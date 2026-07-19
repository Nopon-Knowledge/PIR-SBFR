#!/usr/bin/env python3
"""Forward-only FP16 latency benchmark matching the paper protocol.

Model and optional input construction use delayed ``module:callable`` imports,
so this script does not depend on a particular detector checkpoint wrapper or
post-processing API.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BenchmarkResult:
    device: str
    precision: str
    batch_size: int
    image_size: int
    warmup_passes: int
    runs: int
    passes_per_run: int
    run_mean_latency_ms: Tuple[float, ...]
    median_latency_ms: float
    fps: float
    peak_vram_mib: float
    parameters: int
    trainable_parameters: int

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["run_mean_latency_ms"] = list(self.run_mean_latency_ms)
        result["measurement_scope"] = "network forward only"
        result["excluded"] = [
            "disk access",
            "image decoding",
            "resizing",
            "host-to-device transfer",
            "NMS",
            "output serialization",
        ]
        return result


def resolve_callable(specification: str) -> Callable[..., Any]:
    """Resolve ``package.module:attribute`` without importing model code early."""

    if ":" not in specification:
        raise ValueError(f"expected MODULE:CALLABLE, got {specification!r}")
    module_name, attribute_path = specification.split(":", 1)
    if not module_name or not attribute_path:
        raise ValueError(f"expected MODULE:CALLABLE, got {specification!r}")
    value: Any = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    if not callable(value):
        raise TypeError(f"{specification!r} resolved to a non-callable object")
    return value


def _parse_json_object(raw_value: str, option_name: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{option_name} must decode to a JSON object")
    return value


def _select_checkpoint_value(payload: Any, key: Optional[str]) -> Any:
    if key is not None:
        value = payload
        for component in key.split("."):
            if not isinstance(value, Mapping) or component not in value:
                raise KeyError(f"checkpoint key path {key!r} is absent at {component!r}")
            value = value[component]
        return value
    if isinstance(payload, nn.Module):
        return payload.state_dict()
    if isinstance(payload, Mapping):
        for candidate_key in ("state_dict", "model_state_dict", "model"):
            if candidate_key in payload:
                candidate = payload[candidate_key]
                return candidate.state_dict() if isinstance(candidate, nn.Module) else candidate
    return payload


def load_checkpoint(
    model: nn.Module,
    checkpoint: Path,
    key: Optional[str] = None,
    strict: bool = True,
    allow_pickled_module: bool = False,
) -> None:
    """Load a generic state-dict checkpoint before moving the model to CUDA."""

    payload = torch.load(
        str(checkpoint),
        map_location="cpu",
        weights_only=not allow_pickled_module,
    )
    state_dict = _select_checkpoint_value(payload, key)
    if isinstance(state_dict, nn.Module):
        state_dict = state_dict.state_dict()
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint did not contain a state dict mapping")
    model.load_state_dict(state_dict, strict=strict)


def _move_input(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        dtype = torch.float16 if value.is_floating_point() else value.dtype
        return value.to(device=device, dtype=dtype)
    if isinstance(value, tuple):
        return tuple(_move_input(item, device) for item in value)
    if isinstance(value, list):
        return [_move_input(item, device) for item in value]
    if isinstance(value, Mapping):
        return {key: _move_input(item, device) for key, item in value.items()}
    return value


def _input_tensors(value: Any) -> Sequence[Tensor]:
    if isinstance(value, Tensor):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(tensor for item in value.values() for tensor in _input_tensors(item))
    if isinstance(value, (tuple, list)):
        return tuple(tensor for item in value for tensor in _input_tensors(item))
    return ()


def normalise_forward_inputs(payload: Any, device: torch.device) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Normalize custom input-factory output to ``(*args, **kwargs)``.

    Accepted forms are a tensor, an args tuple/list, a kwargs mapping, or an
    explicit ``(args, kwargs)`` pair.  All tensors are transferred before the
    benchmark, and all floating inputs are converted to FP16.
    """

    if (
        isinstance(payload, tuple)
        and len(payload) == 2
        and isinstance(payload[0], (tuple, list))
        and isinstance(payload[1], Mapping)
    ):
        args = tuple(payload[0])
        kwargs = dict(payload[1])
    elif isinstance(payload, Mapping):
        args = ()
        kwargs = dict(payload)
    elif isinstance(payload, (tuple, list)):
        args = tuple(payload)
        kwargs = {}
    else:
        args = (payload,)
        kwargs = {}

    args = _move_input(args, device)
    kwargs = _move_input(kwargs, device)
    tensors = tuple(_input_tensors(args)) + tuple(_input_tensors(kwargs))
    if not tensors:
        raise ValueError("input factory produced no tensors")
    for tensor in tensors:
        if tensor.ndim > 0 and tensor.shape[0] != 1:
            raise ValueError(f"paper benchmark requires batch size 1, got tensor shape {tuple(tensor.shape)}")
        if tensor.is_floating_point() and tensor.dtype != torch.float16:
            raise ValueError(f"paper benchmark requires FP16 floating inputs, got {tensor.dtype}")
    return tuple(args), dict(kwargs)


ForwardAdapter = Callable[[nn.Module, Tuple[Any, ...], Mapping[str, Any]], Any]


def benchmark_forward(
    model: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    device: torch.device,
    warmup_passes: int = 200,
    runs: int = 5,
    passes_per_run: int = 1_000,
    image_size: int = 640,
    forward_adapter: Optional[ForwardAdapter] = None,
) -> BenchmarkResult:
    """Measure synchronized CUDA forward latency under the paper protocol."""

    if device.type != "cuda":
        raise ValueError("the paper benchmark requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the paper's FP16 latency protocol cannot run")
    if warmup_passes < 0 or runs < 1 or passes_per_run < 1:
        raise ValueError("warmup_passes must be non-negative; runs and passes_per_run must be positive")

    model.to(device=device)
    model.eval()
    model.half()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    def forward_once() -> Any:
        if forward_adapter is None:
            return model(*args, **kwargs)
        return forward_adapter(model, args, kwargs)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    run_means = []
    with torch.inference_mode():
        for _ in range(warmup_passes):
            forward_once()
        torch.cuda.synchronize(device)

        for _ in range(runs):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(passes_per_run):
                forward_once()
            torch.cuda.synchronize(device)
            elapsed_seconds = time.perf_counter() - started
            run_means.append(1_000.0 * elapsed_seconds / passes_per_run)

    latency_ms = float(statistics.median(run_means))
    peak_vram_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
    return BenchmarkResult(
        device=str(device),
        precision="FP16",
        batch_size=1,
        image_size=int(image_size),
        warmup_passes=int(warmup_passes),
        runs=int(runs),
        passes_per_run=int(passes_per_run),
        run_mean_latency_ms=tuple(float(value) for value in run_means),
        median_latency_ms=latency_ms,
        fps=1_000.0 / latency_ms,
        peak_vram_mib=peak_vram_mib,
        parameters=int(total_parameters),
        trainable_parameters=int(trainable_parameters),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FP16 batch-1 forward-only CUDA benchmark")
    parser.add_argument("--factory", required=True, help="model factory as MODULE:CALLABLE")
    parser.add_argument("--factory-kwargs", default="{}", help="JSON object passed to the model factory")
    parser.add_argument("--checkpoint", type=Path, help="optional generic PyTorch state-dict checkpoint")
    parser.add_argument("--checkpoint-key", help="optional dotted path to a state dict inside the checkpoint")
    parser.add_argument("--non-strict", action="store_true", help="load checkpoint with strict=False")
    parser.add_argument(
        "--allow-pickled-module",
        action="store_true",
        help="allow torch.load(weights_only=False); use only for trusted checkpoints",
    )
    parser.add_argument(
        "--input-factory",
        help=(
            "optional MODULE:CALLABLE called with batch_size=1, image_size, device, dtype; "
            "returns a tensor, args, kwargs, or (args, kwargs)"
        ),
    )
    parser.add_argument(
        "--input-factory-kwargs",
        default="{}",
        help="additional JSON object passed to the input factory",
    )
    parser.add_argument(
        "--forward-adapter",
        help="optional MODULE:CALLABLE invoked as adapter(model, args_tuple, kwargs_mapping)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1_000, help="timed forward passes per run")
    parser.add_argument("--output", type=Path, help="optional JSON output path; stdout is always emitted")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.image_size < 1:
        raise ValueError("image size must be positive")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA for the paper benchmark")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the paper's FP16 latency protocol cannot run")

    factory = resolve_callable(args.factory)
    model = factory(**_parse_json_object(args.factory_kwargs, "--factory-kwargs"))
    if not isinstance(model, nn.Module):
        raise TypeError("model factory must return torch.nn.Module")
    if args.checkpoint is not None:
        load_checkpoint(
            model,
            args.checkpoint,
            key=args.checkpoint_key,
            strict=not args.non_strict,
            allow_pickled_module=args.allow_pickled_module,
        )

    if args.input_factory is None:
        input_payload: Any = torch.zeros(
            (1, 3, args.image_size, args.image_size),
            device=device,
            dtype=torch.float16,
        )
    else:
        input_factory = resolve_callable(args.input_factory)
        input_factory_kwargs = _parse_json_object(args.input_factory_kwargs, "--input-factory-kwargs")
        reserved = {"batch_size", "image_size", "device", "dtype"} & set(input_factory_kwargs)
        if reserved:
            raise ValueError(f"input factory kwargs cannot override reserved keys: {sorted(reserved)}")
        input_payload = input_factory(
            batch_size=1,
            image_size=args.image_size,
            device=device,
            dtype=torch.float16,
            **input_factory_kwargs,
        )
    forward_args, forward_kwargs = normalise_forward_inputs(input_payload, device)
    forward_adapter = resolve_callable(args.forward_adapter) if args.forward_adapter else None
    result = benchmark_forward(
        model=model,
        args=forward_args,
        kwargs=forward_kwargs,
        device=device,
        warmup_passes=args.warmup,
        runs=args.runs,
        passes_per_run=args.iterations,
        image_size=args.image_size,
        forward_adapter=forward_adapter,
    )
    rendered = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

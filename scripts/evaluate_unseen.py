#!/usr/bin/env python3
"""Evaluate the nine held-out imaging conditions reported in paper Table 13.

The public implementation fixes the scalar MTF/SNR conversion used for
non-MTF PSFs and non-Poisson noise. The eight single-family conditions follow
the paper protocol directly. The regenerated joint condition is marked
``approximate=true`` because its archived per-image PSF orientation is absent.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pir_sbfr.data.degradations import (
    anisotropic_psf,
    disk_psf,
    motion_psf,
    speckle_noise,
)
from pir_sbfr.evaluation import evaluate_coco
from pir_sbfr.inference import load_checkpoint_model

from _evaluation_common import (
    load_annotation,
    metric_payload,
    run_transformed_coco,
    write_csv,
)


@dataclass(frozen=True)
class OODSpec:
    name: str
    label: str
    family: str
    parameters: Mapping[str, Any]


OOD_SPECS = (
    OODSpec("defocus_psf", "Defocus PSF", "unseen_psf", {"disk_radius_px": 3}),
    OODSpec(
        "motion_psf",
        "Motion PSF",
        "unseen_psf",
        {"length_px": 9, "orientation_degrees": "per-image Uniform[0,180)"},
    ),
    OODSpec(
        "anisotropic_psf",
        "Anisotropic PSF",
        "unseen_psf",
        {"sigma_x": 2.5, "sigma_y": 0.6, "orientation_degrees": "per-image Uniform[0,180)"},
    ),
    OODSpec("speckle_noise", "Speckle noise", "unseen_noise", {"sigma": 0.12}),
    OODSpec(
        "stripe_read_noise",
        "Stripe + read noise",
        "unseen_noise",
        {
            "stripe_amplitude": 0.08,
            "read_sigma": 0.02,
            "implementation": "vertical sinusoid, 4 cycles across width, per-image random phase",
        },
    ),
    OODSpec("sampling_1p5x", "Unseen GSD 1.5x", "unseen_sampling", {"relative_gsd": 1.5}),
    OODSpec("sampling_2p5x", "Unseen GSD 2.5x", "unseen_sampling", {"relative_gsd": 2.5}),
    OODSpec("sampling_4x", "Unseen GSD 4x", "unseen_sampling", {"relative_gsd": 4.0}),
    OODSpec(
        "joint_degradation",
        "Joint degradation",
        "joint_unseen",
        {
            "relative_gsd": 2.5,
            "selected_psf": "motion, length 9, per-image Uniform[0,180) orientation",
            "selected_noise": "multiplicative speckle sigma 0.12",
            "order": ["sampling", "motion_psf", "speckle"],
        },
    ),
)


JOINT_LIMITATION = (
    "The paper specifies the joint operator families but the archived record omits the exact "
    "unseen PSF/noise subtype and sampled PSF orientation. This script generates and records a "
    "deterministic joint realization for the supplied seed; it is not the archived pixel realization."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--dataset", choices=("dior", "aitodv2"), default="dior")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("output/unseen_degradations"))
    parser.add_argument("--category-mapping", type=Path)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser


def _float_image(image: np.ndarray) -> np.ndarray:
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def _sampling_change(image: np.ndarray, ratio: float) -> np.ndarray:
    """Paper sampling: antialiased bicubic reduction, then bicubic restoration."""

    if ratio <= 1.0:
        return _float_image(image)
    source = torch.from_numpy(_float_image(image).transpose(2, 0, 1)).unsqueeze(0)
    height, width = image.shape[:2]
    reduced_size = (max(1, int(round(height / ratio))), max(1, int(round(width / ratio))))
    reduced = F.interpolate(
        source,
        size=reduced_size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    restored = F.interpolate(reduced, size=(height, width), mode="bicubic", align_corners=False)
    return restored.squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0).astype(np.float32)


def _disk_kernel(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    return kernel / kernel.sum()


def _motion_kernel(length: int, angle_degrees: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((length / 2.0 - 0.5, length / 2.0 - 0.5), angle_degrees, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length), flags=cv2.INTER_LINEAR)
    return kernel / max(float(kernel.sum()), np.finfo(np.float32).eps)


def _anisotropic_kernel(sigma_x: float, sigma_y: float, angle_degrees: float) -> np.ndarray:
    radius = int(math.ceil(3.0 * max(sigma_x, sigma_y)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coordinates, coordinates)
    theta = math.radians(angle_degrees)
    xr = xx * math.cos(theta) + yy * math.sin(theta)
    yr = -xx * math.sin(theta) + yy * math.cos(theta)
    kernel = np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2)).astype(np.float32)
    return kernel / kernel.sum()


def _equivalent_nyquist_mtf(kernel: np.ndarray) -> float:
    """Average axial discrete Nyquist response used as reconstructed scalar q."""

    rows = (-1.0) ** np.arange(kernel.shape[0], dtype=np.float32)
    columns = (-1.0) ** np.arange(kernel.shape[1], dtype=np.float32)
    response_y = abs(float((kernel * rows[:, None]).sum()))
    response_x = abs(float((kernel * columns[None, :]).sum()))
    return float(np.clip(0.5 * (response_x + response_y), 1.0e-6, 1.0))


def _measured_snr(clean: np.ndarray, transformed: np.ndarray) -> float:
    source = _float_image(clean)
    result = _float_image(transformed)
    noise_power = float(np.square(result - source).mean())
    signal_power = float(np.square(source).mean())
    if noise_power <= np.finfo(np.float32).eps:
        return 30.0
    return float(10.0 * math.log10(max(signal_power, 1.0e-12) / noise_power))


def _stripe_read(
    image: np.ndarray,
    amplitude: float,
    read_sigma: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    source = _float_image(image)
    width = source.shape[1]
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    stripe = amplitude * np.sin(np.linspace(0.0, 8.0 * math.pi, width, dtype=np.float32) + phase)
    read = rng.normal(0.0, read_sigma, source.shape).astype(np.float32)
    result = np.clip(source + stripe[None, :, None] + read, 0.0, 1.0).astype(np.float32)
    return result, phase


def _apply_spec(
    spec: OODSpec,
    image: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    source = _float_image(image)
    realization: Dict[str, Any] = {}
    if spec.name == "defocus_psf":
        result = disk_psf(source, radius=3)
        realization["equivalent_nyquist_mtf"] = _equivalent_nyquist_mtf(_disk_kernel(3))
    elif spec.name == "motion_psf":
        angle = float(rng.uniform(0.0, 180.0))
        result = motion_psf(source, length=9, angle_degrees=angle)
        realization.update(
            orientation_degrees=angle,
            equivalent_nyquist_mtf=_equivalent_nyquist_mtf(_motion_kernel(9, angle)),
        )
    elif spec.name == "anisotropic_psf":
        angle = float(rng.uniform(0.0, 180.0))
        result = anisotropic_psf(source, sigma_x=2.5, sigma_y=0.6, angle_degrees=angle)
        realization.update(
            orientation_degrees=angle,
            equivalent_nyquist_mtf=_equivalent_nyquist_mtf(_anisotropic_kernel(2.5, 0.6, angle)),
        )
    elif spec.name == "speckle_noise":
        result = speckle_noise(source, sigma=0.12, rng=rng)
        realization["equivalent_snr_db"] = 10.0 * math.log10(1.0 / (0.12**2))
    elif spec.name == "stripe_read_noise":
        result, phase = _stripe_read(source, amplitude=0.08, read_sigma=0.02, rng=rng)
        realization.update(
            phase_radians=phase,
            equivalent_snr_db=_measured_snr(source, result),
        )
    elif spec.name.startswith("sampling_"):
        ratio = float(spec.parameters["relative_gsd"])
        result = _sampling_change(source, ratio)
        realization["relative_gsd"] = ratio
    elif spec.name == "joint_degradation":
        result = _sampling_change(source, 2.5)
        angle = float(rng.uniform(0.0, 180.0))
        result = motion_psf(result, length=9, angle_degrees=angle)
        result = speckle_noise(result, sigma=0.12, rng=rng)
        realization.update(
            relative_gsd=2.5,
            orientation_degrees=angle,
            equivalent_nyquist_mtf=_equivalent_nyquist_mtf(_motion_kernel(9, angle)),
            equivalent_snr_db=10.0 * math.log10(1.0 / (0.12**2)),
        )
    else:  # pragma: no cover - specs are a closed tuple
        raise KeyError(spec.name)
    return np.clip(result, 0.0, 1.0).astype(np.float32), realization


def _descriptor(spec: OODSpec, realization: Mapping[str, Any]) -> Tuple[float, float, float]:
    return (
        float(realization.get("relative_gsd", 1.0)),
        float(realization.get("equivalent_nyquist_mtf", 0.5)),
        float(realization.get("equivalent_snr_db", 30.0)),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch < 1 or args.imgsz < 1:
        raise ValueError("--batch and --imgsz must be positive")
    area_mode = "aitod" if args.dataset == "aitodv2" else "dior"
    max_det = args.max_det if args.max_det is not None else (1500 if area_mode == "aitod" else 100)
    annotation = load_annotation(args.annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint_model(args.weights, args.device)
    condition_results: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for index, spec in enumerate(OOD_SPECS, start=1):
        print(f"OOD {index}/{len(OOD_SPECS)}: {spec.name}", flush=True)
        condition_dir = args.output_dir / spec.name
        predictions_path = condition_dir / "predictions.json"
        routing_path = condition_dir / "routing.json"
        metrics_path = condition_dir / "metrics.json"
        realized: Dict[Any, Dict[str, Any]] = {}

        def transform(image, record, rng, current=spec):
            result, details = _apply_spec(current, image, rng)
            realized[record["id"]] = details
            return result

        def descriptor_provider(record, _path, _clean, _transformed, current=spec):
            return _descriptor(current, realized[record["id"]]), (1.0, 1.0, 1.0)

        predictions, routing = run_transformed_coco(
            model=model,
            annotation=annotation,
            annotations_path=args.annotations,
            images_root=args.images,
            output=predictions_path,
            transform=transform,
            descriptor_provider=descriptor_provider,
            condition_name=spec.name,
            category_mapping=args.category_mapping,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=max_det,
            batch_size=args.batch,
            seed=args.seed,
            routing_output=routing_path,
        )
        for route in routing:
            route["transform_realization"] = realized[route["image_id"]]
        routing_path.write_text(json.dumps(routing, ensure_ascii=False), encoding="utf-8")
        metrics = evaluate_coco(
            args.annotations,
            predictions,
            area_protocol=area_mode,
            max_detections=max_det,
            dior_input_size=args.imgsz if area_mode == "dior" else None,
            quiet=True,
        )
        result: Dict[str, Any] = {
            "index": index,
            "name": spec.name,
            "label": spec.label,
            "family": spec.family,
            "parameters": dict(spec.parameters),
            "approximate": spec.name == "joint_degradation",
            "approximation_note": JOINT_LIMITATION if spec.name == "joint_degradation" else None,
            "metadata_mapping": {
                "gsd": "known sampling ratio, otherwise 1.0 reference",
                "mtf": "mean absolute axial discrete-Nyquist response of the realized PSF, otherwise 0.5 reference",
                "snr": (
                    "10log10(1/sigma^2) for speckle; measured signal/error power "
                    "for stripe+read; otherwise 30 dB reference"
                ),
                "availability": [1, 1, 1],
            },
            "predictions": str(predictions_path),
            "routing": str(routing_path),
        }
        result.update(metric_payload(metrics))
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        condition_results.append(result)
        row: Dict[str, Any] = {
            "index": index,
            "name": spec.name,
            "label": spec.label,
            "family": spec.family,
            "approximate": spec.name == "joint_degradation",
            "metrics_unit": "percent",
        }
        row.update(result["metrics_percent"])
        csv_rows.append(row)

    metric_names = list(condition_results[0]["metrics_percent"]) if condition_results else []
    aggregate: Dict[str, Dict[str, Optional[float]]] = {}
    for metric_name in metric_names:
        values = [
            result["metrics_percent"][metric_name]
            for result in condition_results
            if result["metrics_percent"][metric_name] is not None
        ]
        aggregate[metric_name] = {
            "mean_percent": None if not values else float(np.mean(values)),
            "worst_percent": None if not values else float(np.min(values)),
        }

    summary = {
        "experiment": "paper_table_13_held_out_degradations",
        "approximate_conditions": ["joint_degradation"],
        "approximation_note": JOINT_LIMITATION,
        "dataset": args.dataset,
        "weights": str(args.weights),
        "annotations": str(args.annotations),
        "images": str(args.images),
        "settings": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": max_det,
            "batch": args.batch,
            "seed": args.seed,
            "degradation_stage": "after source-image letterbox to network input",
            "randomization": "BLAKE2b(base seed, condition name, COCO image id)",
        },
        "aggregate": aggregate,
        "conditions": condition_results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "summary.csv", csv_rows)
    print(json.dumps({"summary": str(summary_path), "conditions": len(condition_results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

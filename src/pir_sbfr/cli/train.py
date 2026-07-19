"""Train PIR-SBFR with the exact reported high-level schedule."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import yaml

from pir_sbfr.training import PIRTrainer, paper_train_overrides


def _deep_merge(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("PIR config must contain a YAML mapping")
    base = config.pop("base", None)
    if base:
        base_path = (path.parent / str(base)).resolve()
        return _deep_merge(_load_config(base_path), config)
    return config


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML")
    parser.add_argument("--config", type=Path, default=root / "configs" / "pir_sbfr.yaml")
    parser.add_argument("--scale-mode", choices=("dior", "aitodv2"), default=None)
    parser.add_argument("--seed", type=int, choices=(2023, 2024, 2025), default=2023)
    parser.add_argument("--device", default=None, help="CUDA index, 'mps', or 'cpu'")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/pir_sbfr")
    parser.add_argument("--name", default=None)
    parser.add_argument("--resume", default=False, help="Checkpoint path or false")
    parser.add_argument("--epochs", type=int, default=200, help="Override only for smoke/debug runs")
    parser.add_argument("--batch", type=int, default=16, help="Source images per optimizer update")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-val", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    pir_config = _load_config(args.config)
    pir_config.setdefault("model", {})
    if args.scale_mode:
        pir_config["model"]["scale_mode"] = args.scale_mode
    scale_mode = pir_config["model"].get("scale_mode", "dior")
    run_name = args.name or f"{scale_mode}_seed{args.seed}"
    overrides = paper_train_overrides(
        args.data,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=run_name,
        resume=args.resume,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        nbs=args.batch,
        amp=not args.no_amp,
        val=not args.no_val,
    )
    trainer = PIRTrainer(overrides=overrides, pir_config=pir_config)
    trainer.train()


if __name__ == "__main__":
    main()

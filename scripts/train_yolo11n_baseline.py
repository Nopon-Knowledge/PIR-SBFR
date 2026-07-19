#!/usr/bin/env python3
"""Train the paper's from-scratch YOLO11n reference with the common schedule."""

from __future__ import annotations

import argparse

from ultralytics import YOLO

from pir_sbfr.training import paper_train_overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, choices=(2023, 2024, 2025), default=2023)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/yolo11n")
    parser.add_argument("--name", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    overrides = paper_train_overrides(
        args.data,
        args.seed,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name or f"seed{args.seed}",
        epochs=args.epochs,
        batch=args.batch,
        nbs=args.batch,
        imgsz=args.imgsz,
        amp=not args.no_amp,
    )
    overrides.pop("model", None)
    model = YOLO("yolo11n.yaml")
    model.train(**overrides)


if __name__ == "__main__":
    main()

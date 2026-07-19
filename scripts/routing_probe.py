#!/usr/bin/env python3
"""Probe P3-P5 routing under controlled GSD, MTF, SNR and missing metadata."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from pir_sbfr.data.degradations import DegradationCondition, controlled_degradation
from pir_sbfr.inference import infer_image, letterbox_rgb, load_checkpoint_model


PROBES = (
    ("clear", DegradationCondition(1.0, 0.50, 30.0), (1.0, 1.0, 1.0)),
    ("gsd_3x", DegradationCondition(3.0, 0.50, 30.0), (1.0, 1.0, 1.0)),
    ("mtf_0.15", DegradationCondition(1.0, 0.15, 30.0), (1.0, 1.0, 1.0)),
    ("snr_10", DegradationCondition(1.0, 0.50, 10.0), (1.0, 1.0, 1.0)),
    ("joint", DegradationCondition(3.0, 0.15, 10.0), (1.0, 1.0, 1.0)),
    ("all_missing", DegradationCondition(1.0, 0.50, 30.0), (0.0, 0.0, 0.0)),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output", type=Path, default=Path("output/routing_probe.csv"))
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to decode {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prepared, _ = letterbox_rgb(rgb, args.imgsz)
    model = load_checkpoint_model(args.weights, args.device)
    rows = []
    for index, (name, condition, mask) in enumerate(PROBES):
        degraded = controlled_degradation(prepared, condition, np.random.default_rng(args.seed + index))
        _, aux = infer_image(
            model,
            degraded,
            metadata=condition.metadata,
            availability=mask,
            imgsz=args.imgsz,
        )
        weights = aux["weights"][0].tolist()
        reliability = aux["rho_phy"][0].tolist()
        rows.append(
            {
                "probe": name,
                "gsd": condition.gsd,
                "mtf": condition.mtf,
                "snr": condition.snr,
                "available_gsd": mask[0],
                "available_mtf": mask[1],
                "available_snr": mask[2],
                "w_p3": weights[0],
                "w_p4": weights[1],
                "w_p5": weights[2],
                "rho_p3": reliability[0],
                "rho_p4": reliability[1],
                "rho_p5": reliability[2],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} routing probes to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Report PIR-SBFR parameters and direct-input THOP FLOPs."""

from __future__ import annotations

import argparse
import json

import torch
from thop import profile

from pir_sbfr.models import PIRSBFRModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nc", type=int, default=20, help="number of detector classes")
    parser.add_argument("--imgsz", type=int, default=640, help="square profiling input")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.nc < 1 or args.imgsz < 32:
        raise ValueError("--nc must be positive and --imgsz must be at least 32")
    model = PIRSBFRModel(nc=args.nc).eval()
    sample = torch.zeros(1, 3, args.imgsz, args.imgsz)
    macs, profiled_parameters = profile(model, inputs=(sample,), verbose=False)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "classes": args.nc,
        "input_shape": [1, 3, args.imgsz, args.imgsz],
        "parameters": int(parameters),
        "profiled_parameters": int(profiled_parameters),
        "macs": int(macs),
        "gflops_mac_times_2": float(2.0 * macs / 1.0e9),
        "profiler": "thop direct full-resolution forward",
        "paper_reference_nc20": {"parameters": 3_942_000, "gflops": 8.82},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the paper's three paired seeds in isolated Python processes."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/pir_sbfr.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/pir_sbfr")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    for seed in (2023, 2024, 2025):
        command = [
            sys.executable,
            "-m",
            "pir_sbfr.cli.train",
            "--data",
            args.data,
            "--config",
            args.config,
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--workers",
            str(args.workers),
            "--project",
            args.project,
        ]
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

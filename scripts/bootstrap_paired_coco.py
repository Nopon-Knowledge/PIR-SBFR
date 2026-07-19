#!/usr/bin/env python3
"""Run the paper's image-paired COCO bootstrap protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from pir_sbfr.evaluation.bootstrap import paired_bootstrap_coco


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paired image-level COCO bootstrap. Repeated image IDs are copied "
            "and remapped before AP is recomputed for each replicate."
        )
    )
    parser.add_argument("--annotations", required=True, type=Path, help="COCO ground-truth JSON")
    parser.add_argument("--predictions-a", required=True, type=Path, help="first COCO result JSON (reference)")
    parser.add_argument("--predictions-b", required=True, type=Path, help="second COCO result JSON (candidate)")
    parser.add_argument("--area-mode", choices=("dior", "aitod"), default="dior")
    parser.add_argument("--metric", default="AP", help="paper metric to contrast, e.g. AP, APS, or APVT")
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_718)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--max-detections",
        type=int,
        help="override the protocol cap (DIOR: 100; AI-TOD: 1500)",
    )
    parser.add_argument(
        "--dior-input-size",
        type=int,
        default=640,
        help="letterbox size used to normalize DIOR bbox areas; zero disables normalization",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path; stdout is always emitted")
    parser.add_argument(
        "--include-distribution",
        action="store_true",
        help="include every bootstrap delta in JSON (large for the default 10,000 replicates)",
    )
    parser.add_argument("--progress-every", type=int, default=100, help="progress interval on stderr; zero disables")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    def report_progress(completed: int, total: int) -> None:
        if args.progress_every > 0 and (completed % args.progress_every == 0 or completed == total):
            print(f"bootstrap {completed}/{total}", file=sys.stderr, flush=True)

    result = paired_bootstrap_coco(
        ground_truth=args.annotations,
        predictions_a=args.predictions_a,
        predictions_b=args.predictions_b,
        area_protocol=args.area_mode,
        metric=args.metric,
        replicates=args.replicates,
        seed=args.seed,
        confidence=args.confidence,
        max_detections=args.max_detections,
        dior_input_size=args.dior_input_size or None,
        progress=report_progress,
    )
    rendered = json.dumps(result.to_dict(include_distribution=args.include_distribution), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

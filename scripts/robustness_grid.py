#!/usr/bin/env python3
"""Evaluate one PIR-SBFR checkpoint on the paper's complete 27-cell grid."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pir_sbfr.coco_inference import run_coco_inference
from pir_sbfr.data.degradations import DegradationCondition
from pir_sbfr.evaluation import evaluate_coco
from pir_sbfr.inference import load_checkpoint_model

from _evaluation_common import metric_payload, write_csv


PAPER_GSD = (1.0, 2.0, 3.0)
PAPER_MTF = (0.50, 0.30, 0.15)
PAPER_SNR = (30.0, 20.0, 10.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path, help="COCO ground-truth JSON")
    parser.add_argument("--images", required=True, type=Path, help="image/dataset root")
    parser.add_argument("--weights", required=True, type=Path, help="PIR-SBFR checkpoint")
    parser.add_argument("--dataset", choices=("dior", "aitodv2"), default="dior")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument(
        "--max-det",
        type=int,
        default=None,
        help="NMS/evaluation cap (paper default: DIOR 100, AI-TOD-v2 1500)",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("output/robustness_grid"))
    parser.add_argument("--category-mapping", type=Path)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260718,
        help="paired per-image degradation seed used for every model/cell",
    )
    return parser


def _area_mode(dataset: str) -> str:
    return "aitod" if dataset == "aitodv2" else "dior"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch < 1 or args.imgsz < 1:
        raise ValueError("--batch and --imgsz must be positive")
    area_mode = _area_mode(args.dataset)
    max_det = args.max_det if args.max_det is not None else (1500 if area_mode == "aitod" else 100)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Loading once is important: the 27 cells differ only in the deterministic
    # image-formation condition and use the identical checkpoint state.
    model = load_checkpoint_model(args.weights, args.device)
    cells: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    combinations = tuple(itertools.product(PAPER_GSD, PAPER_MTF, PAPER_SNR))

    for cell_index, (gsd, mtf, snr) in enumerate(combinations, start=1):
        condition = DegradationCondition(gsd=gsd, mtf=mtf, snr=snr)
        cell_dir = args.output_dir / condition.name
        predictions_path = cell_dir / "predictions.json"
        routing_path = cell_dir / "routing.json"
        metrics_path = cell_dir / "metrics.json"
        print(f"grid {cell_index}/{len(combinations)}: {condition.name}", flush=True)
        predictions, _ = run_coco_inference(
            weights=None,
            model=model,
            annotations=args.annotations,
            images_root=args.images,
            output=predictions_path,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=max_det,
            category_mapping=args.category_mapping,
            condition=condition,
            degradation_seed=args.seed,
            routing_output=routing_path,
            batch_size=args.batch,
        )
        metrics = evaluate_coco(
            args.annotations,
            predictions,
            area_protocol=area_mode,
            max_detections=max_det,
            dior_input_size=args.imgsz if area_mode == "dior" else None,
            quiet=True,
        )
        result: Dict[str, Any] = {
            "cell": cell_index,
            "name": condition.name,
            "condition": {"gsd": gsd, "mtf": mtf, "snr": snr},
            "predictions": str(predictions_path),
            "routing": str(routing_path),
        }
        result.update(metric_payload(metrics))
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cells.append(result)
        row: Dict[str, Any] = {
            "cell": cell_index,
            "name": condition.name,
            "gsd": gsd,
            "mtf": mtf,
            "snr_db": snr,
            "metrics_unit": "percent",
        }
        row.update(result["metrics_percent"])
        csv_rows.append(row)

    summary = {
        "experiment": "paper_27_cell_gsd_mtf_snr_grid",
        "dataset": args.dataset,
        "weights": str(args.weights),
        "annotations": str(args.annotations),
        "images": str(args.images),
        "settings": {
            "gsd": list(PAPER_GSD),
            "mtf": list(PAPER_MTF),
            "snr_db": list(PAPER_SNR),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": max_det,
            "batch": args.batch,
            "seed": args.seed,
            "degradation_order": ["sampling", "blur", "shot_noise"],
            "degradation_stage": "after source-image letterbox to network input",
        },
        "cells": cells,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "summary.csv", csv_rows)
    print(json.dumps({"summary": str(summary_path), "cells": len(cells)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

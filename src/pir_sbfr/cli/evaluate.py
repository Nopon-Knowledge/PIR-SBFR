"""Run PIR-SBFR COCO inference and paper-specific DIOR/AI-TOD-v2 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pir_sbfr.coco_inference import run_coco_inference
from pir_sbfr.data.degradations import DegradationCondition
from pir_sbfr.evaluation import evaluate_coco


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--predictions", type=Path, help="Evaluate existing COCO result JSON")
    parser.add_argument("--weights", type=Path, help="Checkpoint; required unless --predictions is supplied")
    parser.add_argument("--images", type=Path, help="Image/dataset root; required with --weights")
    parser.add_argument("--dataset", choices=("dior", "aitodv2"), default="dior")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--category-mapping", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("output/coco_predictions.json"))
    parser.add_argument("--routing-output", type=Path)
    parser.add_argument("--metrics-output", type=Path, default=Path("output/metrics.json"))
    parser.add_argument("--gsd", type=float, default=None, help="Controlled condition relative GSD")
    parser.add_argument("--mtf", type=float, default=None, help="Controlled condition Nyquist MTF")
    parser.add_argument("--snr", type=float, default=None, help="Controlled condition SNR dB")
    parser.add_argument("--degradation-seed", type=int, default=20260718)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    area_mode = "aitod" if args.dataset == "aitodv2" else "dior"
    max_det = args.max_det or (1500 if area_mode == "aitod" else 100)
    predictions_path = args.predictions

    controlled_values = (args.gsd, args.mtf, args.snr)
    if any(value is not None for value in controlled_values):
        condition = DegradationCondition(
            gsd=1.0 if args.gsd is None else args.gsd,
            mtf=0.5 if args.mtf is None else args.mtf,
            snr=30.0 if args.snr is None else args.snr,
        )
    else:
        condition = None

    if predictions_path is None:
        if args.weights is None or args.images is None:
            raise ValueError("--weights and --images are required unless --predictions is supplied")
        run_coco_inference(
            weights=args.weights,
            annotations=args.annotations,
            images_root=args.images,
            output=args.output,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=max_det,
            metadata_json=args.metadata,
            category_mapping=args.category_mapping,
            condition=condition,
            degradation_seed=args.degradation_seed,
            routing_output=args.routing_output,
            batch_size=args.batch,
        )
        predictions_path = args.output

    metrics = evaluate_coco(
        args.annotations,
        predictions_path,
        area_protocol=area_mode,
        max_detections=max_det,
        dior_input_size=args.imgsz if area_mode == "dior" else None,
        quiet=True,
    )
    result = {
        "dataset": args.dataset,
        "predictions": str(predictions_path),
        "condition": None if condition is None else condition.__dict__,
        "metrics": metrics,
        "metrics_percent": {key: (None if value < 0 else 100.0 * value) for key, value in metrics.items()},
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

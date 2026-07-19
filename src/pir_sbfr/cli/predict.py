"""Run metadata-aware PIR-SBFR inference on an image or directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from pir_sbfr.inference import (
    find_metadata_record,
    image_paths,
    infer_image,
    load_checkpoint_model,
    load_metadata_records,
    record_acquisition,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--metadata", help="JSON mapping path/name/stem to gsd/mtf/snr/availability")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("output/predictions.json"))
    parser.add_argument("--routing-output", type=Path, default=None)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    model = load_checkpoint_model(args.weights, args.device)
    records = load_metadata_records(args.metadata)
    predictions = []
    routing = []
    paths = image_paths(args.source)
    if not paths:
        raise RuntimeError("no supported images found")

    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to decode {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        values, mask = record_acquisition(find_metadata_record(records, path))
        detections, aux = infer_image(
            model,
            rgb,
            metadata=values,
            availability=mask,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
        )
        for x1, y1, x2, y2, confidence, class_id in detections.tolist():
            predictions.append(
                {
                    "image": str(path),
                    "class_id": int(class_id),
                    "class_name": model.names.get(int(class_id), str(int(class_id))),
                    "confidence": float(confidence),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                }
            )
        routing.append(
            {
                "image": str(path),
                "metadata": list(values),
                "availability": list(mask),
                "weights": aux["weights"][0].tolist(),
                "rho_phy": aux["rho_phy"][0].tolist(),
                "scale_estimate": aux["scale_estimate"][0].tolist(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False, indent=2)
    if args.routing_output:
        args.routing_output.parent.mkdir(parents=True, exist_ok=True)
        with args.routing_output.open("w", encoding="utf-8") as handle:
            json.dump(routing, handle, ensure_ascii=False, indent=2)
    print(f"saved {len(predictions)} detections for {len(paths)} images to {args.output}")


if __name__ == "__main__":
    main()

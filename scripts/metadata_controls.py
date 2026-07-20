#!/usr/bin/env python3
"""Run the complete Table 12 metadata-correspondence controls on one COCO set.

For the closest reproduction, provide the already mixed-degraded images and
their per-image descriptors with ``--metadata``.  The paper's mixed control set
was not released; ``--synthesize-controlled`` creates a clearly labelled,
deterministic approximation from clean images instead.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pir_sbfr.data.degradations import DegradationCondition, controlled_degradation
from pir_sbfr.evaluation import evaluate_coco
from pir_sbfr.inference import (
    find_metadata_record,
    load_checkpoint_model,
    load_metadata_records,
    record_acquisition,
)

from _evaluation_common import (
    load_annotation,
    metric_payload,
    resolve_image,
    run_transformed_coco,
    stable_seed,
    write_csv,
)


REFERENCE = (1.0, 0.5, 30.0)
CONTROL_NAMES = (
    "correct",
    "multiplicative_error_10pct",
    "multiplicative_error_20pct",
    "multiplicative_error_50pct",
    "missing_25pct",
    "missing_50pct",
    "missing_100pct",
    "training_set_mean",
    "cross_image_shuffled",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--metadata",
        "--base-metadata",
        dest="metadata",
        type=Path,
        help="correct per-image metadata JSON for the already degraded images",
    )
    source.add_argument(
        "--synthesize-controlled",
        action="store_true",
        help="construct an approximate mixed-degradation set from the input clean images",
    )
    parser.add_argument(
        "--training-mean",
        type=float,
        nargs=3,
        metavar=("GSD", "MTF", "SNR"),
        help="training-set mean descriptor; absent uses a labelled evaluation-set proxy",
    )
    parser.add_argument("--dataset", choices=("dior", "aitodv2"), default="dior")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("output/metadata_controls"))
    parser.add_argument("--category-mapping", type=Path)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser


def _identity(image: np.ndarray, _record, _rng) -> np.ndarray:
    return image


def _sample_synthetic_conditions(
    records: Sequence[Mapping[str, Any]], seed: int
) -> Dict[Any, DegradationCondition]:
    grid = [
        DegradationCondition(gsd, mtf, snr)
        for gsd, mtf, snr in itertools.product(
            (1.0, 2.0, 3.0), (0.5, 0.3, 0.15), (30.0, 20.0, 10.0)
        )
        if (gsd, mtf, snr) != REFERENCE
    ]
    sampled: Dict[Any, DegradationCondition] = {}
    for record in records:
        rng = np.random.default_rng(stable_seed(seed, "mixed_condition", record["id"]))
        sampled[record["id"]] = grid[int(rng.integers(0, len(grid)))]
    return sampled


def _load_base_descriptors(
    annotation: Mapping[str, Any],
    images_root: Path,
    metadata_path: Path,
) -> Dict[Any, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    metadata = load_metadata_records(metadata_path)
    descriptors = {}
    missing = []
    for record in annotation.get("images", []):
        path = resolve_image(images_root, str(record["file_name"]))
        source = find_metadata_record(metadata, path)
        if source is None:
            missing.append(str(record["file_name"]))
            continue
        values, mask = record_acquisition(source)
        descriptors[record["id"]] = (tuple(values), tuple(mask))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"base metadata has no record for {len(missing)} COCO images (first: {preview})"
        )
    return descriptors


def _available_mean(
    descriptors: Mapping[Any, Tuple[Sequence[float], Sequence[float]]]
) -> Tuple[float, float, float]:
    result = []
    for field in range(3):
        values = [
            float(descriptor[0][field])
            for descriptor in descriptors.values()
            if float(descriptor[1][field]) > 0.0
        ]
        if not values:
            raise ValueError(f"cannot estimate metadata mean: field {field} is always unavailable")
        result.append(float(np.mean(values)))
    return tuple(result)  # type: ignore[return-value]


def _clamp_descriptor(values: Sequence[float]) -> Tuple[float, float, float]:
    return (
        max(float(values[0]), 1.0e-6),
        min(max(float(values[1]), 1.0e-6), 1.0),
        max(float(values[2]), 1.0e-6),
    )


def _build_control_descriptors(
    base: Mapping[Any, Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    training_mean: Sequence[float],
    seed: int,
) -> Dict[str, Dict[Any, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]]:
    variants: Dict[
        str, Dict[Any, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]
    ] = {"correct": dict(base)}

    for percent in (10, 20, 50):
        name = f"multiplicative_error_{percent}pct"
        fraction = percent / 100.0
        controls = {}
        for image_id, (values, mask) in base.items():
            rng = np.random.default_rng(stable_seed(seed, name, image_id))
            factors = 1.0 + rng.uniform(-fraction, fraction, size=3)
            controls[image_id] = (
                _clamp_descriptor(np.asarray(values, dtype=np.float64) * factors),
                mask,
            )
        variants[name] = controls

    for percent in (25, 50, 100):
        name = f"missing_{percent}pct"
        fraction = percent / 100.0
        controls = {}
        for image_id, (values, mask) in base.items():
            if percent == 100:
                retained = np.zeros(3, dtype=np.float64)
            else:
                rng = np.random.default_rng(stable_seed(seed, name, image_id))
                retained = (rng.random(3) >= fraction).astype(np.float64)
            new_mask = tuple(float(left) * float(right) for left, right in zip(mask, retained))
            controls[image_id] = (values, new_mask)
        variants[name] = controls

    constant = _clamp_descriptor(training_mean)
    variants["training_set_mean"] = {
        image_id: (constant, (1.0, 1.0, 1.0)) for image_id in base
    }

    image_ids = list(base)
    if len(image_ids) < 2:
        raise ValueError("cross-image shuffled metadata requires at least two images")
    rng = np.random.default_rng(stable_seed(seed, "cross_image_shuffled", len(image_ids)))
    order = list(np.asarray(image_ids, dtype=object)[rng.permutation(len(image_ids))])
    source_for_target = {
        order[position]: order[(position + 1) % len(order)] for position in range(len(order))
    }
    variants["cross_image_shuffled"] = {
        target: base[source_for_target[target]] for target in image_ids
    }
    return variants


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch < 1 or args.imgsz < 1:
        raise ValueError("--batch and --imgsz must be positive")
    area_mode = "aitod" if args.dataset == "aitodv2" else "dior"
    max_det = args.max_det if args.max_det is not None else (1500 if area_mode == "aitod" else 100)
    annotation = load_annotation(args.annotations)
    records = annotation.get("images", [])

    synthetic_conditions: Dict[Any, DegradationCondition] = {}
    if args.synthesize_controlled:
        synthetic_conditions = _sample_synthetic_conditions(records, args.seed)
        base = {
            record["id"]: (synthetic_conditions[record["id"]].metadata, (1.0, 1.0, 1.0))
            for record in records
        }
        source_mode = "reconstructed_synthetic_mixed_degradation"
    else:
        assert args.metadata is not None
        base = _load_base_descriptors(annotation, args.images, args.metadata)
        source_mode = "user_supplied_already_degraded_images_and_metadata"

    if args.training_mean is None:
        training_mean = _available_mean(base)
        training_mean_source = "evaluation-set available-field mean proxy"
    else:
        training_mean = _clamp_descriptor(args.training_mean)
        training_mean_source = "user-supplied training-set mean"
    variants = _build_control_descriptors(base, training_mean, args.seed)

    if args.synthesize_controlled:
        def transform(image, record, rng):
            return controlled_degradation(image, synthetic_conditions[record["id"]], rng)
    else:
        transform = _identity

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint_model(args.weights, args.device)
    results: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    correct_metrics_percent: Optional[Dict[str, Optional[float]]] = None

    for index, name in enumerate(CONTROL_NAMES, start=1):
        print(f"metadata control {index}/{len(CONTROL_NAMES)}: {name}", flush=True)
        condition_dir = args.output_dir / name
        predictions_path = condition_dir / "predictions.json"
        routing_path = condition_dir / "routing.json"
        metrics_path = condition_dir / "metrics.json"
        descriptors = variants[name]

        def descriptor_provider(record, _path, _clean, _transformed, current=descriptors):
            return current[record["id"]]

        predictions, _ = run_transformed_coco(
            model=model,
            annotation=annotation,
            annotations_path=args.annotations,
            images_root=args.images,
            output=predictions_path,
            transform=transform,
            descriptor_provider=descriptor_provider,
            # A constant label keeps synthetic Poisson realizations identical
            # while only descriptor values/masks change across controls.
            condition_name="mixed_degradation_images",
            category_mapping=args.category_mapping,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=max_det,
            batch_size=args.batch,
            seed=args.seed,
            routing_output=routing_path,
        )
        metrics = evaluate_coco(
            args.annotations,
            predictions,
            area_protocol=area_mode,
            max_detections=max_det,
            dior_input_size=args.imgsz if area_mode == "dior" else None,
            quiet=True,
        )
        payload = metric_payload(metrics)
        metrics_percent = payload["metrics_percent"]
        if correct_metrics_percent is None:
            correct_metrics_percent = dict(metrics_percent)
        delta_pp = {
            metric: (
                None
                if value is None or correct_metrics_percent.get(metric) is None
                else float(value) - float(correct_metrics_percent[metric])
            )
            for metric, value in metrics_percent.items()
        }
        result: Dict[str, Any] = {
            "index": index,
            "condition": name,
            "source_mode": source_mode,
            "approximate": True,
            "predictions": str(predictions_path),
            "routing": str(routing_path),
            "within_set_delta_pp_vs_correct": delta_pp,
        }
        result.update(payload)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        results.append(result)
        row: Dict[str, Any] = {
            "index": index,
            "condition": name,
            "metrics_unit": "percent",
        }
        row.update({f"{key}_percent": value for key, value in metrics_percent.items()})
        row.update({f"delta_{key}_pp": value for key, value in delta_pp.items()})
        csv_rows.append(row)

    approximation_notes = []
    if args.synthesize_controlled:
        approximation_notes.append(
            "The paper's separate mixed-degradation image list and per-image settings are not public. "
            "This run assigns each source image uniformly to one of the 26 non-reference controlled-grid cells."
        )
    if args.training_mean is None:
        approximation_notes.append(
            "No training-set descriptor mean was supplied; the available-field mean of this "
            "evaluation set is used as a proxy."
        )
    approximation_notes.append(
        "The PDF reports perturbation magnitudes but not random draws; multiplicative errors are "
        "independent Uniform[-p,+p] per field, missing masks are independent Bernoulli draws, "
        "and shuffling is a seeded cyclic derangement."
    )
    summary = {
        "experiment": "paper_table_10_within_set_metadata_controls",
        "source_mode": source_mode,
        "approximate": True,
        "approximation_notes": approximation_notes,
        "dataset": args.dataset,
        "weights": str(args.weights),
        "annotations": str(args.annotations),
        "images": str(args.images),
        "base_metadata": None if args.metadata is None else str(args.metadata),
        "training_mean": list(training_mean),
        "training_mean_source": training_mean_source,
        "settings": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": max_det,
            "batch": args.batch,
            "seed": args.seed,
            "degradation_stage": (
                "already present in user-provided images"
                if not args.synthesize_controlled
                else "controlled transform after source-image letterbox to network input"
            ),
            "comparison": "within-set percentage-point delta from correct metadata only",
        },
        "conditions": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "summary.csv", csv_rows)
    print(json.dumps({"summary": str(summary_path), "conditions": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

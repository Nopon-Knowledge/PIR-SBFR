#!/usr/bin/env python3
"""Shared, script-local helpers for reconstructed COCO stress tests.

The paper applies synthetic image formation after the source image has been
letterboxed to the network input size.  The public inference helper normally
performs that letterbox internally, so the experiment scripts use this small
runner to keep the order explicit and to restore detections to source-image
coordinates before COCO evaluation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from pir_sbfr.coco_inference import class_to_category_map
from pir_sbfr.inference import infer_images, letterbox_rgb, restore_boxes


Descriptor = Tuple[Sequence[float], Sequence[float]]
Transform = Callable[[np.ndarray, Mapping[str, Any], np.random.Generator], np.ndarray]
DescriptorProvider = Callable[[Mapping[str, Any], Path, np.ndarray, np.ndarray], Descriptor]


def stable_seed(base_seed: int, condition: str, image_id: Any) -> int:
    """Return a process-independent uint64 seed for one image/condition."""

    payload = f"{int(base_seed)}|{condition}|{image_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little")


def load_annotation(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError(f"{path} is not a COCO annotation object")
    return payload


def resolve_image(images_root: Path, file_name: str) -> Path:
    """Resolve common COCO path layouts without silently choosing duplicates."""

    relative = Path(str(file_name).replace("\\", "/"))
    candidates = [images_root / relative, images_root / relative.name]
    if len(relative.parts) > 1:
        candidates.append(images_root / Path(*relative.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = list(images_root.rglob(relative.name))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ValueError(f"ambiguous COCO image {file_name!r} below {images_root}")
    raise FileNotFoundError(f"cannot resolve COCO image {file_name!r} below {images_root}")


def as_model_rgb(image: np.ndarray) -> np.ndarray:
    """Validate a transform result without requantizing normalized floats."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("transform must return an HWC RGB image")
    if np.issubdtype(image.dtype, np.integer):
        return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or float(image.min()) < 0.0 or float(image.max()) > 1.0:
            raise ValueError("floating transform results must be finite and normalized to [0,1]")
        return np.ascontiguousarray(image.astype(np.float32))
    raise TypeError(f"unsupported transform dtype: {image.dtype}")


def run_transformed_coco(
    model: Any,
    annotation: Mapping[str, Any],
    annotations_path: Path,
    images_root: Path,
    output: Path,
    transform: Transform,
    descriptor_provider: DescriptorProvider,
    condition_name: str,
    category_mapping: Optional[Path],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    batch_size: int,
    seed: int,
    routing_output: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Infer transformed, already-letterboxed inputs and emit COCO records."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    category_ids = class_to_category_map(annotation, category_mapping)
    predictions: List[Dict[str, Any]] = []
    routing: List[Dict[str, Any]] = []
    records = annotation.get("images", [])

    for start in range(0, len(records), batch_size):
        record_batch = records[start : start + batch_size]
        network_images: List[np.ndarray] = []
        original_infos = []
        paths: List[Path] = []
        clean_prepared: List[np.ndarray] = []
        transformed_prepared: List[np.ndarray] = []

        for record in record_batch:
            path = resolve_image(images_root, str(record["file_name"]))
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"failed to decode {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prepared, info = letterbox_rgb(rgb, imgsz)
            rng = np.random.default_rng(stable_seed(seed, condition_name, record["id"]))
            transformed = as_model_rgb(transform(prepared, record, rng))
            network_images.append(transformed)
            original_infos.append(info)
            paths.append(path)
            clean_prepared.append(prepared)
            transformed_prepared.append(transformed)

        values_batch: List[Sequence[float]] = []
        masks_batch: List[Sequence[float]] = []
        for record, path, clean, transformed in zip(record_batch, paths, clean_prepared, transformed_prepared):
            values, mask = descriptor_provider(record, path, clean, transformed)
            if len(values) != 3 or len(mask) != 3:
                raise ValueError("descriptor values and availability masks must each have length three")
            values_batch.append(tuple(float(value) for value in values))
            masks_batch.append(tuple(float(value) for value in mask))

        detections_batch, aux = infer_images(
            model,
            network_images,
            values_batch,
            masks_batch,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
        )
        for index, (record, detections, info) in enumerate(zip(record_batch, detections_batch, original_infos)):
            detections = detections.clone()
            if detections.numel():
                detections[:, :4] = restore_boxes(detections[:, :4], info)
            for x1, y1, x2, y2, score, class_id_float in detections.tolist():
                class_id = int(class_id_float)
                if class_id not in category_ids:
                    raise KeyError(f"model class {class_id} has no COCO category mapping")
                predictions.append(
                    {
                        "image_id": record["id"],
                        "category_id": category_ids[class_id],
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    }
                )
            routing.append(
                {
                    "image_id": record["id"],
                    "file_name": record["file_name"],
                    "metadata": list(values_batch[index]),
                    "availability": list(masks_batch[index]),
                    "weights": aux["weights"][index].tolist(),
                    "rho_phy": aux["rho_phy"][index].tolist(),
                    "scale_estimate": aux["scale_estimate"][index].tolist(),
                }
            )
        completed = min(start + batch_size, len(records))
        if completed % 100 == 0 or completed == len(records):
            print(f"{condition_name}: inference {completed}/{len(records)}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    if routing_output is not None:
        routing_output.parent.mkdir(parents=True, exist_ok=True)
        routing_output.write_text(json.dumps(routing, ensure_ascii=False), encoding="utf-8")
    return predictions, routing


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_payload(metrics: Mapping[str, float]) -> Dict[str, Any]:
    return {
        "metrics": dict(metrics),
        "metrics_percent": {key: (None if value < 0.0 else 100.0 * float(value)) for key, value in metrics.items()},
    }


__all__ = [
    "as_model_rgb",
    "load_annotation",
    "metric_payload",
    "resolve_image",
    "run_transformed_coco",
    "stable_seed",
    "write_csv",
]

"""Run PIR-SBFR over a COCO image list and emit standard result records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from pir_sbfr.data.degradations import DegradationCondition, controlled_degradation
from pir_sbfr.inference import (
    find_metadata_record,
    infer_images,
    letterbox_rgb,
    load_checkpoint_model,
    load_metadata_records,
    record_acquisition,
    restore_boxes,
)


def _resolve_coco_image(images_root: Path, file_name: str) -> Path:
    relative = Path(file_name.replace("\\", "/"))
    candidates = (
        images_root / relative,
        images_root / relative.name,
        images_root / Path(*relative.parts[1:]) if len(relative.parts) > 1 else images_root / relative,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = list(images_root.rglob(relative.name))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ValueError(f"ambiguous COCO image {file_name!r} below {images_root}")
    raise FileNotFoundError(f"cannot resolve COCO image {file_name!r} below {images_root}")


def class_to_category_map(annotation: Mapping, mapping_path: Optional[Path] = None) -> Dict[int, int]:
    if mapping_path:
        with mapping_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping):
            if "mapping" in payload:
                payload = payload["mapping"]
            elif "categories" in payload:
                payload = payload["categories"]
        if isinstance(payload, list):
            result = {}
            for item in payload:
                yolo_id = item.get("yolo_id", item.get("class_id"))
                coco_id = item.get("coco_id", item.get("category_id"))
                if yolo_id is None or coco_id is None:
                    raise ValueError("category mapping entries require yolo_id and coco_id")
                result[int(yolo_id)] = int(coco_id)
            return result
        if isinstance(payload, Mapping):
            return {int(key): int(value) for key, value in payload.items()}
        raise ValueError("unsupported category mapping JSON structure")
    categories = sorted(annotation.get("categories", []), key=lambda item: int(item["id"]))
    return {index: int(category["id"]) for index, category in enumerate(categories)}


def _image_seed(base_seed: int, image_id) -> int:
    digest = hashlib.blake2b(f"{base_seed}|{image_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def run_coco_inference(
    weights: Optional[Path],
    annotations: Path,
    images_root: Path,
    output: Path,
    device: Optional[str] = None,
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.70,
    max_det: int = 1500,
    metadata_json: Optional[Path] = None,
    category_mapping: Optional[Path] = None,
    condition: Optional[DegradationCondition] = None,
    degradation_seed: int = 20260718,
    routing_output: Optional[Path] = None,
    batch_size: int = 16,
    model=None,
) -> Tuple[List[Dict], List[Dict]]:
    """Infer in COCO annotation order, optionally under one controlled cell."""
    with annotations.open("r", encoding="utf-8-sig") as handle:
        ground_truth = json.load(handle)
    if model is None:
        if weights is None:
            raise ValueError("weights are required when no preloaded model is supplied")
        model = load_checkpoint_model(weights, device)
    metadata_records = load_metadata_records(metadata_json)
    category_ids = class_to_category_map(ground_truth, category_mapping)
    predictions: List[Dict] = []
    routing: List[Dict] = []

    image_records = ground_truth.get("images", [])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(image_records), batch_size):
        records_batch = image_records[start : start + batch_size]
        images_batch = []
        values_batch = []
        masks_batch = []
        controlled_restore_infos = []
        for image_record in records_batch:
            image_id = image_record["id"]
            path = _resolve_coco_image(images_root, image_record["file_name"])
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"failed to decode {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if condition is None:
                values, mask = record_acquisition(find_metadata_record(metadata_records, path))
                network_image = rgb
                controlled_restore_infos.append(None)
            else:
                # Section 3.5.1 defines x as the already resized 640x640
                # network input. Preserve the first letterbox transform so the
                # resulting detections can still be reported in source pixels.
                prepared, source_info = letterbox_rgb(rgb, imgsz)
                transformed = controlled_degradation(
                    prepared,
                    condition,
                    np.random.default_rng(_image_seed(degradation_seed, image_id)),
                )
                # Preserve the paper's [0,1] floating image formation result.
                network_image = np.ascontiguousarray(transformed.astype(np.float32))
                values, mask = condition.metadata, (1.0, 1.0, 1.0)
                controlled_restore_infos.append(source_info)
            images_batch.append(network_image)
            values_batch.append(values)
            masks_batch.append(mask)

        detections_batch, aux = infer_images(
            model,
            images_batch,
            values_batch,
            masks_batch,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
        )
        for local_index, (image_record, detections) in enumerate(zip(records_batch, detections_batch)):
            image_id = image_record["id"]
            source_info = controlled_restore_infos[local_index]
            if source_info is not None and detections.numel():
                detections = detections.clone()
                detections[:, :4] = restore_boxes(detections[:, :4], source_info)
            for x1, y1, x2, y2, score, class_id_float in detections.tolist():
                class_id = int(class_id_float)
                if class_id not in category_ids:
                    raise KeyError(f"model class {class_id} has no COCO category mapping")
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_ids[class_id],
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    }
                )
            routing.append(
                {
                    "image_id": image_id,
                    "file_name": image_record["file_name"],
                    "metadata": list(values_batch[local_index]),
                    "availability": list(masks_batch[local_index]),
                    "weights": aux["weights"][local_index].tolist(),
                    "rho_phy": aux["rho_phy"][local_index].tolist(),
                    "scale_estimate": aux["scale_estimate"][local_index].tolist(),
                }
            )
        completed = min(start + batch_size, len(image_records))
        if completed % 100 == 0 or completed == len(image_records):
            print(f"inference {completed}/{len(image_records)}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False)
    if routing_output:
        routing_output.parent.mkdir(parents=True, exist_ok=True)
        with routing_output.open("w", encoding="utf-8") as handle:
            json.dump(routing, handle, ensure_ascii=False)
    return predictions, routing


__all__ = ["class_to_category_map", "run_coco_inference"]

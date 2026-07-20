"""Convert AI-TOD-v2 COCO annotations to a YOLO training layout."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Mapping, Optional

from .common import (
    MATERIALIZE_MODES,
    labels_text,
    require_file,
    resolve_input_path,
    safe_materialize,
    safe_relative_path,
    safe_write_json,
    safe_write_text,
    safe_write_yaml,
    yolo_line,
)


@dataclass(frozen=True)
class CocoImageRecord:
    image_id: Hashable
    file_name: str
    output_relative: Path
    source: Path
    width: int
    height: int


@dataclass(frozen=True)
class CocoSplit:
    name: str
    annotation_path: Path
    payload: dict[str, Any]
    images: tuple[CocoImageRecord, ...]


@dataclass(frozen=True)
class PreparedCocoSplit:
    split: CocoSplit
    lines_by_id: Mapping[Hashable, tuple[str, ...]]
    kept_annotations: int
    skipped_crowd: int
    skipped_degenerate: int


def _load_json(path: Path) -> dict[str, Any]:
    path = require_file(path, "AI-TOD-v2 COCO annotation JSON")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"COCO annotation root must be an object: {path}")
    for key in ("images", "annotations"):
        if not isinstance(payload.get(key, []), list):
            raise ValueError(f"COCO field {key!r} must be a list: {path}")
    if not isinstance(payload.get("categories", []), list):
        raise ValueError(f"COCO field 'categories' must be a list: {path}")
    return payload


def _normalize_output_relative(file_name: str, split: str) -> Path:
    relative = safe_relative_path(file_name, "COCO image file_name")
    parts = list(relative.parts)
    if parts and parts[0].casefold() == "images":
        parts.pop(0)
    if parts and parts[0].casefold() == split.casefold():
        parts.pop(0)
    if not parts:
        raise ValueError(f"COCO file_name has no filename component: {file_name!r}")
    return Path(*parts)


def _resolve_image(
    source_root: Path,
    split: str,
    file_name: str,
    explicit_image_dir: Optional[Path],
) -> Path:
    relative = safe_relative_path(file_name, "COCO image file_name")
    candidates: list[Path] = []
    if explicit_image_dir is not None:
        candidates.extend((explicit_image_dir / relative, explicit_image_dir / relative.name))
    candidates.extend(
        (
            source_root / relative,
            source_root / "images" / relative,
            source_root / split / relative,
            source_root / "images" / split / relative,
            source_root / split / relative.name,
            source_root / "images" / split / relative.name,
        )
    )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"Cannot resolve AI-TOD-v2 image {file_name!r} for split {split!r}; "
        f"checked under {explicit_image_dir or source_root}"
    )


def _validate_identifier(value: Any, description: str) -> Hashable:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{description} must be an integer or string, got {value!r}")
    return value


def _parse_split(
    name: str,
    annotation_path: Path,
    source_root: Path,
    image_dir: Optional[Path],
) -> CocoSplit:
    payload = _load_json(annotation_path)
    images: list[CocoImageRecord] = []
    seen_ids: set[Hashable] = set()
    seen_outputs: dict[Path, str] = {}
    seen_labels: dict[Path, str] = {}
    for position, raw in enumerate(payload.get("images", []), 1):
        if not isinstance(raw, dict):
            raise ValueError(f"Image entry {position} in {annotation_path} must be an object")
        image_id = _validate_identifier(raw.get("id"), f"Image id at position {position} in {annotation_path}")
        if image_id in seen_ids:
            raise ValueError(f"Duplicate image id {image_id!r} in {annotation_path}")
        seen_ids.add(image_id)
        file_name = raw.get("file_name")
        if not isinstance(file_name, str):
            raise ValueError(f"Image {image_id!r} in {annotation_path} has no string file_name")
        width, height = raw.get("width"), raw.get("height")
        if isinstance(width, bool) or isinstance(height, bool):
            raise ValueError(f"Invalid size for image {image_id!r} in {annotation_path}")
        try:
            width, height = int(width), int(height)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid size for image {image_id!r} in {annotation_path}") from error
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid size ({width}, {height}) for image {image_id!r} in {annotation_path}")
        output_relative = _normalize_output_relative(file_name, name)
        if output_relative in seen_outputs:
            raise ValueError(
                f"COCO file_names {seen_outputs[output_relative]!r} and {file_name!r} map to the same output "
                f"path in split {name!r}"
            )
        seen_outputs[output_relative] = file_name
        label_relative = output_relative.with_suffix(".txt")
        if label_relative in seen_labels:
            raise ValueError(
                f"COCO file_names {seen_labels[label_relative]!r} and {file_name!r} map to the same YOLO label "
                f"path in split {name!r}"
            )
        seen_labels[label_relative] = file_name
        images.append(
            CocoImageRecord(
                image_id=image_id,
                file_name=file_name,
                output_relative=output_relative,
                source=_resolve_image(source_root, name, file_name, image_dir),
                width=width,
                height=height,
            )
        )
    return CocoSplit(name=name, annotation_path=annotation_path, payload=payload, images=tuple(images))


def _category_mapping(splits: Mapping[str, CocoSplit]) -> tuple[list[dict[str, Any]], dict[int, int]]:
    category_names: dict[int, str] = {}
    for split in splits.values():
        for position, category in enumerate(split.payload.get("categories", []), 1):
            if not isinstance(category, dict):
                raise ValueError(f"Category entry {position} in {split.annotation_path} must be an object")
            category_id = category.get("id")
            if isinstance(category_id, bool) or not isinstance(category_id, int):
                raise ValueError(f"COCO category id must be an integer in {split.annotation_path}: {category_id!r}")
            name = category.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"COCO category {category_id!r} has no valid name in {split.annotation_path}")
            if category_id in category_names and category_names[category_id] != name:
                raise ValueError(
                    f"COCO category id {category_id} is named both {category_names[category_id]!r} and {name!r}"
                )
            category_names[category_id] = name
    if not category_names:
        raise ValueError("No COCO categories were found in any supplied AI-TOD-v2 annotation file")

    ordered = [
        {"coco_id": category_id, "yolo_id": yolo_id, "name": category_names[category_id]}
        for yolo_id, category_id in enumerate(sorted(category_names))
    ]
    return ordered, {entry["coco_id"]: entry["yolo_id"] for entry in ordered}


def _prepare_annotations(
    split: CocoSplit,
    coco_to_yolo: Mapping[int, int],
    include_crowd: bool,
) -> PreparedCocoSplit:
    by_id = {image.image_id: image for image in split.images}
    lines_by_id: dict[Hashable, list[str]] = {image.image_id: [] for image in split.images}
    skipped_crowd = 0
    skipped_degenerate = 0
    kept_annotations = 0
    for position, annotation in enumerate(split.payload.get("annotations", []), 1):
        if not isinstance(annotation, dict):
            raise ValueError(f"Annotation entry {position} in {split.annotation_path} must be an object")
        image_id = _validate_identifier(
            annotation.get("image_id"), f"Annotation image_id at position {position} in {split.annotation_path}"
        )
        if image_id not in by_id:
            raise ValueError(
                f"Annotation {position} references unknown image id {image_id!r} in {split.annotation_path}"
            )
        category_id = annotation.get("category_id")
        if isinstance(category_id, bool) or not isinstance(category_id, int) or category_id not in coco_to_yolo:
            raise ValueError(
                f"Annotation {position} references unknown category id {category_id!r} in {split.annotation_path}"
            )
        if bool(annotation.get("iscrowd", 0)) and not include_crowd:
            skipped_crowd += 1
            continue
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Annotation {position} has an invalid COCO bbox in {split.annotation_path}")
        try:
            x, y, width, height = (float(value) for value in bbox)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Annotation {position} has a non-numeric bbox in {split.annotation_path}") from error
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(f"Annotation {position} has a non-finite bbox in {split.annotation_path}")
        image = by_id[image_id]
        x0 = min(float(image.width), max(0.0, x))
        y0 = min(float(image.height), max(0.0, y))
        x1 = min(float(image.width), max(0.0, x + width))
        y1 = min(float(image.height), max(0.0, y + height))
        clipped_width, clipped_height = x1 - x0, y1 - y0
        if clipped_width <= 0 or clipped_height <= 0:
            skipped_degenerate += 1
            continue
        lines_by_id[image_id].append(
            yolo_line(
                coco_to_yolo[category_id],
                (
                    (x0 + clipped_width / 2.0) / image.width,
                    (y0 + clipped_height / 2.0) / image.height,
                    clipped_width / image.width,
                    clipped_height / image.height,
                ),
            )
        )
        kept_annotations += 1
    return PreparedCocoSplit(
        split=split,
        lines_by_id={image_id: tuple(lines) for image_id, lines in lines_by_id.items()},
        kept_annotations=kept_annotations,
        skipped_crowd=skipped_crowd,
        skipped_degenerate=skipped_degenerate,
    )


def prepare_aitodv2(
    source_root: Path,
    output_root: Path,
    annotation_files: Mapping[str, Path],
    image_dirs: Optional[Mapping[str, Path]] = None,
    mode: str = "symlink",
    include_crowd: bool = False,
) -> dict[str, Any]:
    """Convert one or more AI-TOD-v2 COCO splits into YOLO labels.

    COCO category ids are sorted numerically and mapped to contiguous zero-based
    YOLO ids.  The exact mapping is saved in ``category_mapping.json`` and the
    original COCO JSON for each split is copied byte-for-byte into ``annotations``.
    """

    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"AI-TOD-v2 source root does not exist: {source_root}")
    if mode not in MATERIALIZE_MODES:
        raise ValueError(f"mode must be one of {MATERIALIZE_MODES}, got {mode!r}")
    if not annotation_files:
        raise ValueError("At least one AI-TOD-v2 annotation JSON must be supplied")
    unsupported = set(annotation_files) - {"train", "val", "test"}
    if unsupported:
        raise ValueError(f"Unsupported AI-TOD-v2 splits: {sorted(unsupported)}")

    resolved_image_dirs: dict[str, Path] = {}
    for split, directory in (image_dirs or {}).items():
        resolved = resolve_input_path(Path(directory), source_root)
        if not resolved.is_dir():
            raise NotADirectoryError(f"AI-TOD-v2 image directory does not exist for {split}: {resolved}")
        resolved_image_dirs[split] = resolved

    splits: dict[str, CocoSplit] = {}
    for split in ("train", "val", "test"):
        if split not in annotation_files:
            continue
        annotation_path = resolve_input_path(Path(annotation_files[split]), source_root)
        splits[split] = _parse_split(split, annotation_path, source_root, resolved_image_dirs.get(split))

    mapping_entries, coco_to_yolo = _category_mapping(splits)
    mapping_payload = {
        "ordering": "ascending COCO category id",
        "categories": mapping_entries,
        "coco_to_yolo": {str(entry["coco_id"]): entry["yolo_id"] for entry in mapping_entries},
        "yolo_to_coco": {str(entry["yolo_id"]): entry["coco_id"] for entry in mapping_entries},
    }
    # Validate and convert every annotation before creating any destination files.
    prepared_splits = {
        split_name: _prepare_annotations(split, coco_to_yolo, include_crowd) for split_name, split in splits.items()
    }

    summary: dict[str, Any] = {
        "dataset": "AI-TOD-v2",
        "mode": mode,
        "include_crowd": include_crowd,
        "categories": mapping_entries,
        "splits": {},
    }
    for split_name, prepared in prepared_splits.items():
        split = prepared.split
        for image in split.images:
            safe_materialize(image.source, output_root / "images" / split_name / image.output_relative, mode)
            label_relative = image.output_relative.with_suffix(".txt")
            safe_write_text(
                output_root / "labels" / split_name / label_relative,
                labels_text(prepared.lines_by_id[image.image_id]),
            )
        safe_materialize(split.annotation_path, output_root / "annotations" / f"{split_name}.json", "copy")
        summary["splits"][split_name] = {
            "source_annotation": str(split.annotation_path),
            "images": len(split.images),
            "annotations": prepared.kept_annotations,
            "skipped_crowd": prepared.skipped_crowd,
            "skipped_degenerate": prepared.skipped_degenerate,
        }

    dataset_yaml: dict[str, Any] = {"path": str(output_root)}
    for split in ("train", "val", "test"):
        if split in splits:
            dataset_yaml[split] = f"images/{split}"
    dataset_yaml["names"] = [entry["name"] for entry in mapping_entries]
    safe_write_yaml(output_root / "dataset.yaml", dataset_yaml)
    safe_write_json(output_root / "category_mapping.json", mapping_payload)
    safe_write_json(output_root / "conversion_report.json", summary)
    return summary

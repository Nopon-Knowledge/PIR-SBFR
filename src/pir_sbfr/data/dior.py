"""Convert the official DIOR Pascal-VOC release to YOLO and COCO formats."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from PIL import Image

from .common import (
    labels_text,
    read_id_file,
    require_file,
    resolve_input_path,
    safe_materialize,
    safe_write_json,
    safe_write_text,
    safe_write_yaml,
    yolo_line,
)


# This is the official DIOR order used by the paper and must not be alphabetized.
DIOR_CLASSES: tuple[str, ...] = (
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
)

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def _class_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


_CLASS_TO_INDEX = {_class_key(name): index for index, name in enumerate(DIOR_CLASSES)}


@dataclass(frozen=True)
class DiorObject:
    category_index: int
    bbox: tuple[float, float, float, float]
    difficult: int
    truncated: int


@dataclass(frozen=True)
class DiorImage:
    image_id: str
    source: Path
    output_name: str
    width: int
    height: int
    objects: tuple[DiorObject, ...]


def _xml_text(node: ET.Element, path: str, default: Optional[str] = None) -> str:
    value = node.findtext(path)
    if value is None or not value.strip():
        if default is not None:
            return default
        raise ValueError(f"Required XML field {path!r} is missing")
    return value.strip()


def _find_image(images_dir: Path, image_id: str, xml_filename: Optional[str]) -> Path:
    candidates: list[Path] = []
    if xml_filename:
        # Some DIOR mirrors store a stale absolute path; only the filename is relevant.
        candidates.append(images_dir / Path(xml_filename.replace("\\", "/")).name)
    id_path = Path(image_id)
    if id_path.suffix:
        candidates.append(images_dir / id_path.name)
    else:
        for suffix in _IMAGE_SUFFIXES:
            candidates.extend((images_dir / f"{image_id}{suffix}", images_dir / f"{image_id}{suffix.upper()}"))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate.resolve()

    # Fall back to a case-insensitive extension check for unusual but valid mirrors.
    matches = sorted(
        path
        for path in images_dir.glob(f"{id_path.stem}.*")
        if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
    )
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ValueError(f"Ambiguous image files for DIOR id {image_id!r}: {matches}")
    raise FileNotFoundError(f"No image found for DIOR id {image_id!r} under {images_dir}")


def _parse_xml(
    xml_path: Path,
    images_dir: Path,
    image_id: str,
    one_based_inclusive: bool,
    exclude_difficult: bool,
) -> DiorImage:
    try:
        root = ET.parse(require_file(xml_path, "DIOR annotation XML")).getroot()
    except ET.ParseError as error:
        raise ValueError(f"Invalid XML in {xml_path}: {error}") from error

    source = _find_image(images_dir, image_id, root.findtext("filename"))
    width_text = root.findtext("size/width")
    height_text = root.findtext("size/height")
    if width_text and height_text:
        width, height = int(float(width_text)), int(float(height_text))
    else:
        with Image.open(source) as image:
            width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size ({width}, {height}) in {xml_path}")

    objects: list[DiorObject] = []
    for object_number, obj in enumerate(root.findall("object"), 1):
        class_name = _xml_text(obj, "name")
        class_key = _class_key(class_name)
        if class_key not in _CLASS_TO_INDEX:
            raise ValueError(
                f"Unknown DIOR class {class_name!r} in {xml_path} object {object_number}; "
                f"expected one of {DIOR_CLASSES}"
            )
        difficult = int(float(_xml_text(obj, "difficult", "0")))
        truncated = int(float(_xml_text(obj, "truncated", "0")))
        if exclude_difficult and difficult:
            continue

        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bndbox in {xml_path} object {object_number}")
        xmin = float(_xml_text(box, "xmin"))
        ymin = float(_xml_text(box, "ymin"))
        xmax = float(_xml_text(box, "xmax"))
        ymax = float(_xml_text(box, "ymax"))
        if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
            raise ValueError(f"Non-finite bbox in {xml_path} object {object_number}")

        # Official VOC XML coordinates are 1-based and inclusive.  Converting them
        # to a zero-based half-open interval also makes a [1, width] box span 100%.
        if one_based_inclusive:
            xmin -= 1.0
            ymin -= 1.0
        xmin = min(float(width), max(0.0, xmin))
        ymin = min(float(height), max(0.0, ymin))
        xmax = min(float(width), max(0.0, xmax))
        ymax = min(float(height), max(0.0, ymax))
        box_width, box_height = xmax - xmin, ymax - ymin
        if box_width <= 0 or box_height <= 0:
            raise ValueError(
                f"Degenerate bbox ({xmin}, {ymin}, {xmax}, {ymax}) in {xml_path} object {object_number}"
            )
        objects.append(
            DiorObject(
                category_index=_CLASS_TO_INDEX[class_key],
                bbox=(xmin, ymin, box_width, box_height),
                difficult=difficult,
                truncated=truncated,
            )
        )

    output_name = f"{Path(image_id).stem}{source.suffix.lower()}"
    return DiorImage(
        image_id=image_id,
        source=source,
        output_name=output_name,
        width=width,
        height=height,
        objects=tuple(objects),
    )


def _coco_document(split: str, images: list[DiorImage]) -> dict[str, Any]:
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for numeric_image_id, image in enumerate(images, 1):
        coco_images.append(
            {
                "id": numeric_image_id,
                "file_name": f"images/{split}/{image.output_name}",
                "width": image.width,
                "height": image.height,
                "dior_id": image.image_id,
            }
        )
        for obj in image.objects:
            x, y, width, height = obj.bbox
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": numeric_image_id,
                    "category_id": obj.category_index + 1,
                    "bbox": [round(x, 6), round(y, 6), round(width, 6), round(height, 6)],
                    "area": round(width * height, 6),
                    "segmentation": [],
                    # Stock COCOeval derives its ignore mask from iscrowd and
                    # otherwise overwrites the generic ``ignore`` field.
                    "iscrowd": int(bool(obj.difficult)),
                    "ignore": int(bool(obj.difficult)),
                    "difficult": obj.difficult,
                    "truncated": obj.truncated,
                }
            )
            annotation_id += 1
    return {
        "info": {
            "description": "DIOR ground truth converted from the official Pascal-VOC annotations",
            "version": "1.0",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": index + 1, "name": name, "supercategory": "object"}
            for index, name in enumerate(DIOR_CLASSES)
        ],
    }


def prepare_dior(
    source_root: Path,
    output_root: Path,
    split_id_files: Optional[Mapping[str, Path]] = None,
    images_dir: Optional[Path] = None,
    split_image_dirs: Optional[Mapping[str, Path]] = None,
    annotations_dir: Path = Path("Annotations"),
    splits_dir: Optional[Path] = None,
    mode: str = "symlink",
    one_based_inclusive: bool = True,
    exclude_difficult: bool = False,
) -> dict[str, Any]:
    """Prepare DIOR for YOLO training and COCO evaluation.

    ``split_id_files`` defaults to the official
    ``ImageSets/Main/{train,val,test}.txt`` files.  All inputs are validated before
    output files are created.
    """

    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"DIOR source root does not exist: {source_root}")
    annotations_dir = resolve_input_path(annotations_dir, source_root)
    if not annotations_dir.is_dir():
        raise NotADirectoryError(f"DIOR annotations directory does not exist: {annotations_dir}")
    if splits_dir is None:
        split_candidates = (source_root / "ImageSets/Main", source_root / "ImageSets")
        splits_dir = next((candidate for candidate in split_candidates if candidate.is_dir()), split_candidates[0])
    else:
        splits_dir = resolve_input_path(splits_dir, source_root)

    if split_id_files is None:
        split_paths = {split: splits_dir / f"{split}.txt" for split in ("train", "val", "test")}
    else:
        if not split_id_files:
            raise ValueError("At least one DIOR split must be supplied")
        split_paths = {
            split: resolve_input_path(Path(path), source_root) for split, path in split_id_files.items()
        }
    unsupported = set(split_paths) - {"train", "val", "test"}
    if unsupported:
        raise ValueError(f"Unsupported DIOR splits: {sorted(unsupported)}")

    common_images_dir = resolve_input_path(images_dir, source_root) if images_dir is not None else None
    supplied_image_dirs = split_image_dirs or {}
    unsupported_image_splits = set(supplied_image_dirs) - {"train", "val", "test"}
    if unsupported_image_splits:
        raise ValueError(f"Unsupported DIOR image-directory splits: {sorted(unsupported_image_splits)}")
    image_dirs: dict[str, Path] = {}
    for split in split_paths:
        if split in supplied_image_dirs:
            image_dirs[split] = resolve_input_path(Path(supplied_image_dirs[split]), source_root)
        elif common_images_dir is not None:
            image_dirs[split] = common_images_dir
        else:
            if split in {"train", "val"}:
                names = (
                    "JPEGImages-trainval",
                    "JPEGImages_trainval",
                    "Images_trainval",
                    "JPEGImages",
                    "trainval",
                    split,
                )
            else:
                names = ("JPEGImages-test", "JPEGImages_test", "Images_test", "JPEGImages", "test")
            candidates = tuple(source_root / name for name in names)
            image_dirs[split] = next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])
        if not image_dirs[split].is_dir():
            raise NotADirectoryError(f"DIOR images directory does not exist for {split}: {image_dirs[split]}")

    parsed: dict[str, list[DiorImage]] = {}
    owner_by_id: dict[str, str] = {}
    for split in ("train", "val", "test"):
        if split not in split_paths:
            continue
        ids = read_id_file(split_paths[split])
        parsed[split] = []
        for image_id in ids:
            if image_id in owner_by_id:
                raise ValueError(
                    f"DIOR image id {image_id!r} occurs in both {owner_by_id[image_id]!r} and {split!r} splits"
                )
            owner_by_id[image_id] = split
            xml_path = annotations_dir / f"{Path(image_id).stem}.xml"
            parsed[split].append(
                _parse_xml(xml_path, image_dirs[split], image_id, one_based_inclusive, exclude_difficult)
            )

    summary: dict[str, Any] = {
        "dataset": "DIOR",
        "mode": mode,
        "class_names": list(DIOR_CLASSES),
        "voc_coordinate_convention": "one-based-inclusive" if one_based_inclusive else "zero-based-half-open",
        "exclude_difficult": exclude_difficult,
        "splits": {},
    }
    for split, images in parsed.items():
        object_count = 0
        for image in images:
            safe_materialize(image.source, output_root / "images" / split / image.output_name, mode)
            label_lines: list[str] = []
            for obj in image.objects:
                x, y, width, height = obj.bbox
                label_lines.append(
                    yolo_line(
                        obj.category_index,
                        (
                            (x + width / 2.0) / image.width,
                            (y + height / 2.0) / image.height,
                            width / image.width,
                            height / image.height,
                        ),
                    )
                )
            safe_write_text(
                output_root / "labels" / split / f"{Path(image.output_name).stem}.txt",
                labels_text(label_lines),
            )
            object_count += len(image.objects)
        coco = _coco_document(split, images)
        safe_write_json(output_root / "annotations" / f"{split}.json", coco)
        summary["splits"][split] = {"images": len(images), "annotations": object_count}

    dataset_yaml: dict[str, Any] = {"path": str(output_root)}
    for split in ("train", "val", "test"):
        if split in parsed:
            dataset_yaml[split] = f"images/{split}"
    dataset_yaml["names"] = list(DIOR_CLASSES)
    safe_write_yaml(output_root / "dataset.yaml", dataset_yaml)
    safe_write_json(output_root / "conversion_report.json", summary)
    return summary

#!/usr/bin/env python3
"""Prepare the official DIOR dataset for PIR-SBFR training/evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DIOR Pascal-VOC XML and official train/val/test id files to "
            "YOLO labels plus COCO ground truth. Existing conflicting outputs are never overwritten."
        )
    )
    parser.add_argument(
        "--source",
        "--source-root",
        dest="source",
        type=Path,
        required=True,
        help="DIOR root containing JPEGImages, Annotations, and ImageSets/Main",
    )
    parser.add_argument(
        "--output",
        "--output-root",
        dest="output",
        type=Path,
        required=True,
        help="New/reusable YOLO dataset root",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="One image directory for every split; default auto-detects JPEGImages or the official split folders",
    )
    for split in ("train", "val", "test"):
        parser.add_argument(
            f"--{split}-images-dir",
            type=Path,
            help=f"Override the image directory for {split}, relative to --source",
        )
    parser.add_argument(
        "--annotations-dir", type=Path, default=Path("Annotations"), help="VOC XML directory, relative to --source"
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        help="Directory containing train.txt, val.txt, test.txt; default auto-detects ImageSets/Main or ImageSets",
    )
    parser.add_argument("--train-ids", type=Path, help="Override official train id file (relative to --source)")
    parser.add_argument("--val-ids", type=Path, help="Override official validation id file (relative to --source)")
    parser.add_argument("--test-ids", type=Path, help="Override official test id file (relative to --source)")
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Materialize images as relative symlinks (default) or byte copies",
    )
    parser.add_argument(
        "--voc-coordinates",
        choices=("one-based-inclusive", "zero-based-half-open"),
        default="one-based-inclusive",
        help="Interpretation of XML bbox coordinates (official DIOR uses the default)",
    )
    parser.add_argument(
        "--exclude-difficult",
        action="store_true",
        help="Omit VOC objects marked difficult from both YOLO and generated COCO ground truth",
    )
    return parser


def _existing_path_or_source_relative(path: Path) -> Path:
    """Accept both paths relative to the current directory and to --source."""

    expanded = path.expanduser()
    if expanded.is_absolute() or expanded.exists():
        return expanded.resolve()
    return expanded


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    repository_src = Path(__file__).resolve().parents[1] / "src"
    if str(repository_src) not in sys.path:
        sys.path.insert(0, str(repository_src))
    from pir_sbfr.data.dior import prepare_dior

    source = args.source.expanduser().resolve()
    if args.splits_dir is None:
        split_candidates = (source / "ImageSets/Main", source / "ImageSets")
        splits_dir = next((candidate for candidate in split_candidates if candidate.is_dir()), split_candidates[0])
    else:
        splits_dir = _existing_path_or_source_relative(args.splits_dir)
    split_files = {}
    split_image_dirs = {}
    for split in ("train", "val", "test"):
        id_path = getattr(args, f"{split}_ids") or splits_dir / f"{split}.txt"
        split_files[split] = _existing_path_or_source_relative(id_path)
        image_dir = getattr(args, f"{split}_images_dir")
        if image_dir is not None:
            split_image_dirs[split] = _existing_path_or_source_relative(image_dir)
    summary = prepare_dior(
        source_root=source,
        output_root=args.output,
        split_id_files=split_files,
        images_dir=_existing_path_or_source_relative(args.images_dir) if args.images_dir is not None else None,
        split_image_dirs=split_image_dirs,
        annotations_dir=_existing_path_or_source_relative(args.annotations_dir),
        splits_dir=splits_dir,
        mode=args.mode,
        one_based_inclusive=args.voc_coordinates == "one-based-inclusive",
        exclude_difficult=args.exclude_difficult,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

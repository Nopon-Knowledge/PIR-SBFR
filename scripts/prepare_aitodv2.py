#!/usr/bin/env python3
"""Prepare AI-TOD-v2 COCO data for PIR-SBFR training/evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


_JSON_CANDIDATES = {
    "train": (
        "aitodv2_train.json",
        "aitod_training.json",
        "aitodv2_training.json",
        "aitod_train_v2.json",
        "instances_train.json",
        "train.json",
    ),
    "val": (
        "aitodv2_val.json",
        "aitod_validation.json",
        "aitodv2_validation.json",
        "aitod_val_v2.json",
        "instances_val.json",
        "val.json",
    ),
    "test": (
        "aitodv2_test.json",
        "aitod_test.json",
        "aitod_testing.json",
        "aitod_test_v2.json",
        "instances_test.json",
        "test.json",
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert AI-TOD-v2 COCO JSON to YOLO labels while retaining the original JSON and an explicit "
            "COCO-to-YOLO category mapping. Existing conflicting outputs are never overwritten."
        )
    )
    parser.add_argument(
        "--source",
        "--source-root",
        dest="source",
        type=Path,
        required=True,
        help="AI-TOD-v2 root containing annotations and images",
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
        "--annotations-dir",
        type=Path,
        default=Path("annotations"),
        help="Directory searched for common official JSON filenames, relative to --source",
    )
    for split in ("train", "val", "test"):
        parser.add_argument(
            f"--{split}-json",
            type=Path,
            help=f"COCO JSON for {split} (relative to --source); overrides filename discovery",
        )
        parser.add_argument(
            f"--{split}-images-dir",
            type=Path,
            help=f"Optional image directory for {split}, relative to --source",
        )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Materialize images as relative symlinks (default) or byte copies",
    )
    parser.add_argument(
        "--include-crowd",
        action="store_true",
        help="Include annotations marked iscrowd=1 in YOLO labels (default: retain only in copied COCO GT)",
    )
    return parser


def _discover_json(source: Path, annotations_dir: Path, split: str) -> Optional[Path]:
    search_dir = annotations_dir if annotations_dir.is_absolute() else source / annotations_dir
    for filename in _JSON_CANDIDATES[split]:
        candidate = search_dir / filename
        if candidate.is_file():
            return candidate
    return None


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
    from pir_sbfr.data.aitodv2 import prepare_aitodv2

    source = args.source.expanduser().resolve()
    annotation_files = {}
    image_dirs = {}
    for split in ("train", "val", "test"):
        annotation_path = getattr(args, f"{split}_json") or _discover_json(source, args.annotations_dir, split)
        if annotation_path is not None:
            annotation_files[split] = _existing_path_or_source_relative(annotation_path)
        image_dir = getattr(args, f"{split}_images_dir")
        if image_dir is not None:
            image_dirs[split] = _existing_path_or_source_relative(image_dir)
    if not annotation_files:
        searched = args.annotations_dir if args.annotations_dir.is_absolute() else source / args.annotations_dir
        _parser().error(
            f"no annotation JSON found under {searched}; pass at least one of "
            "--train-json/--val-json/--test-json"
        )

    summary = prepare_aitodv2(
        source_root=source,
        output_root=args.output,
        annotation_files=annotation_files,
        image_dirs=image_dirs,
        mode=args.mode,
        include_crowd=args.include_crowd,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

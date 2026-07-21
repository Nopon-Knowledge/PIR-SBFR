"""Evaluation utilities for the open-source PIR-SBFR implementation."""

from .bootstrap import BootstrapResult, RemappedCocoSample, paired_bootstrap_coco, remap_paired_coco_sample
from .coco import (
    AITOD_AREA_PROTOCOL,
    DIOR_AREA_PROTOCOL,
    AreaProtocol,
    PaperCOCOeval,
    configure_area_ranges,
    evaluate_coco,
    get_area_protocol,
)

__all__ = [
    "AITOD_AREA_PROTOCOL",
    "DIOR_AREA_PROTOCOL",
    "AreaProtocol",
    "BootstrapResult",
    "PaperCOCOeval",
    "RemappedCocoSample",
    "configure_area_ranges",
    "evaluate_coco",
    "get_area_protocol",
    "paired_bootstrap_coco",
    "remap_paired_coco_sample",
]

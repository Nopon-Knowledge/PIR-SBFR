"""COCO evaluation configured for the paper's object-size protocols.

The stock :class:`pycocotools.cocoeval.COCOeval` hard-codes the three COCO
area labels in ``summarize``.  That is sufficient for DIOR after changing its
two thresholds, but it cannot report AI-TOD-v2's four size intervals.  This
module keeps the official matching and accumulation implementation and only
replaces area configuration and summary extraction.
"""

from __future__ import annotations

import copy
import io
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


COCO_MAX_AREA = 1.0e10


@dataclass(frozen=True)
class AreaProtocol:
    """Named COCO area ranges and their paper metric suffixes.

    ``COCOeval`` treats both endpoints as inclusive, which is the convention
    used by the reference evaluator.  Consequently, an object exactly on a
    shared boundary is included by both neighbouring diagnostic summaries;
    overall AP is unaffected.
    """

    name: str
    labels: Tuple[str, ...]
    ranges: Tuple[Tuple[float, float], ...]
    metric_suffixes: Tuple[str, ...]
    reference_image_size: Optional[int] = None
    max_detections: Tuple[int, int, int] = (1, 10, 100)

    def __post_init__(self) -> None:
        if not (len(self.labels) == len(self.ranges) == len(self.metric_suffixes)):
            raise ValueError("labels, ranges, and metric_suffixes must have equal length")
        if not self.labels or self.labels[0] != "all" or self.metric_suffixes[0] != "":
            raise ValueError("the first area range must be the unsuffixed 'all' range")


DIOR_AREA_PROTOCOL = AreaProtocol(
    name="dior",
    labels=("all", "small", "medium", "large"),
    ranges=(
        (0.0, COCO_MAX_AREA),
        (0.0, float(32**2)),
        (float(32**2), float(96**2)),
        (float(96**2), COCO_MAX_AREA),
    ),
    metric_suffixes=("", "S", "M", "L"),
    reference_image_size=640,
    max_detections=(1, 10, 100),
)


AITOD_AREA_PROTOCOL = AreaProtocol(
    name="aitod",
    labels=("all", "verytiny", "tiny", "small", "medium"),
    ranges=(
        (0.0, COCO_MAX_AREA),
        (float(2**2), float(8**2)),
        (float(8**2), float(16**2)),
        (float(16**2), float(32**2)),
        (float(32**2), float(64**2)),
    ),
    metric_suffixes=("", "VT", "T", "S", "M"),
    reference_image_size=None,
    max_detections=(1, 100, 1500),
)


_PROTOCOL_ALIASES = {
    "dior": DIOR_AREA_PROTOCOL,
    "aitod": AITOD_AREA_PROTOCOL,
    "ai-tod": AITOD_AREA_PROTOCOL,
    "aitod-v2": AITOD_AREA_PROTOCOL,
    "ai-tod-v2": AITOD_AREA_PROTOCOL,
}


def get_area_protocol(protocol: Union[str, AreaProtocol]) -> AreaProtocol:
    """Resolve a paper area protocol by name."""

    if isinstance(protocol, AreaProtocol):
        return protocol
    key = str(protocol).strip().lower().replace("_", "-")
    try:
        return _PROTOCOL_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROTOCOL_ALIASES))
        raise ValueError(f"unknown area protocol {protocol!r}; choose one of: {choices}") from exc


def configure_area_ranges(coco_eval: COCOeval, protocol: Union[str, AreaProtocol]) -> AreaProtocol:
    """Apply the paper's area ranges to an existing ``COCOeval`` instance."""

    resolved = get_area_protocol(protocol)
    coco_eval.params.areaRng = [list(bounds) for bounds in resolved.ranges]
    coco_eval.params.areaRngLbl = list(resolved.labels)
    return resolved


class PaperCOCOeval(COCOeval):
    """Official COCO matching/accumulation with paper-specific area summaries."""

    def __init__(
        self,
        coco_gt: Optional[COCO] = None,
        coco_dt: Optional[COCO] = None,
        iou_type: str = "bbox",
        area_protocol: Union[str, AreaProtocol] = "dior",
    ) -> None:
        super().__init__(coco_gt, coco_dt, iou_type)
        self.area_protocol = configure_area_ranges(self, area_protocol)
        self.params.maxDets = list(self.area_protocol.max_detections)
        self.metric_names: Tuple[str, ...] = ()
        self.metrics: Dict[str, float] = {}

    @staticmethod
    def _valid_mean(values: np.ndarray) -> float:
        valid = values[values > -1]
        return float(np.mean(valid)) if valid.size else -1.0

    def _summarize_value(
        self,
        ap: bool,
        area_label: str = "all",
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> float:
        if not self.eval:
            raise RuntimeError("run evaluate() and accumulate() before summarizing")

        max_detections = self.params.maxDets[-1] if max_detections is None else int(max_detections)
        area_indices = [index for index, label in enumerate(self.params.areaRngLbl) if label == area_label]
        max_det_indices = [index for index, value in enumerate(self.params.maxDets) if value == max_detections]
        if not area_indices:
            raise KeyError(f"area label {area_label!r} is not configured")
        if not max_det_indices:
            raise KeyError(f"maxDets={max_detections} is not configured")

        if ap:
            # precision dimensions: IoU x recall x category x area x maxDets
            values = self.eval["precision"]
            if iou_threshold is not None:
                indices = np.flatnonzero(np.isclose(self.params.iouThrs, iou_threshold))
                values = values[indices]
            values = values[:, :, :, area_indices, max_det_indices]
        else:
            # recall dimensions: IoU x category x area x maxDets
            values = self.eval["recall"]
            if iou_threshold is not None:
                indices = np.flatnonzero(np.isclose(self.params.iouThrs, iou_threshold))
                values = values[indices]
            values = values[:, :, area_indices, max_det_indices]
        return self._valid_mean(values)

    def summary_metrics(self) -> Dict[str, float]:
        """Return DIOR or AI-TOD metrics using the notation in the paper."""

        max_det = self.params.maxDets[-1]
        metrics: Dict[str, float] = {
            "AP": self._summarize_value(True, max_detections=max_det),
            "AP50": self._summarize_value(True, iou_threshold=0.50, max_detections=max_det),
            "AP75": self._summarize_value(True, iou_threshold=0.75, max_detections=max_det),
        }
        for label, suffix in zip(self.area_protocol.labels[1:], self.area_protocol.metric_suffixes[1:]):
            metrics[f"AP{suffix}"] = self._summarize_value(True, area_label=label, max_detections=max_det)

        for diagnostic_max_det in self.params.maxDets[:-1]:
            metrics[f"AR{diagnostic_max_det}"] = self._summarize_value(
                False,
                max_detections=diagnostic_max_det,
            )
        metrics["AR"] = self._summarize_value(False, max_detections=max_det)
        for label, suffix in zip(self.area_protocol.labels[1:], self.area_protocol.metric_suffixes[1:]):
            metrics[f"AR{suffix}"] = self._summarize_value(False, area_label=label, max_detections=max_det)
        return metrics

    def summarize(self) -> Dict[str, float]:
        """Print and store a summary without COCO's hard-coded area labels."""

        metrics = self.summary_metrics()
        self.metrics = metrics
        self.metric_names = tuple(metrics)
        self.stats = np.asarray(tuple(metrics.values()), dtype=np.float64)
        for name, value in metrics.items():
            print(f"{name:<5s} = {value:0.3f}")
        return dict(metrics)


CocoSource = Union[str, Path, Mapping[str, Any], COCO]
PredictionSource = Union[str, Path, Sequence[Mapping[str, Any]]]


def _normalise_ground_truth(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(dataset))
    result.setdefault("info", {})
    result.setdefault("licenses", [])
    result.setdefault("images", [])
    result.setdefault("categories", [])
    annotations = result.setdefault("annotations", [])
    for index, annotation in enumerate(annotations, start=1):
        annotation.setdefault("id", index)
        annotation.setdefault("iscrowd", 0)
        if "area" not in annotation:
            bbox = annotation.get("bbox")
            if bbox is None or len(bbox) != 4:
                raise ValueError("every ground-truth annotation needs 'area' or a four-value 'bbox'")
            annotation["area"] = float(bbox[2]) * float(bbox[3])
    return result


def load_coco_ground_truth(source: CocoSource, quiet: bool = True) -> COCO:
    """Load a COCO object from a path or an in-memory mapping."""

    if isinstance(source, COCO):
        return source
    if isinstance(source, (str, Path)):
        output_context = redirect_stdout(io.StringIO()) if quiet else nullcontext()
        with output_context:
            return COCO(str(source))

    coco = COCO()
    coco.dataset = _normalise_ground_truth(source)
    output_context = redirect_stdout(io.StringIO()) if quiet else nullcontext()
    with output_context:
        coco.createIndex()
    return coco


def load_coco_predictions(source: PredictionSource) -> Sequence[Mapping[str, Any]]:
    """Load a COCO result list, leaving records immutable to the caller."""

    if isinstance(source, (str, Path)):
        import json

        with Path(source).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, Mapping) and "annotations" in data:
            data = data["annotations"]
    else:
        data = source
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        raise TypeError("COCO predictions must be a JSON array or a sequence of mappings")
    return data


def _letterbox_area_scales(coco_gt: COCO, image_size: int) -> Dict[Any, float]:
    """Return per-image area factors for square, aspect-preserving letterbox."""

    if image_size < 1:
        raise ValueError("reference image size must be positive")
    scales: Dict[Any, float] = {}
    for image in coco_gt.dataset.get("images", []):
        image_id = image.get("id")
        width = image.get("width")
        height = image.get("height")
        if width is None or height is None:
            raise ValueError(f"DIOR area normalization needs width and height for image_id={image_id!r}")
        if float(width) <= 0.0 or float(height) <= 0.0:
            raise ValueError(f"image_id={image_id!r} has invalid dimensions {width!r} x {height!r}")
        gain = min(float(image_size) / float(width), float(image_size) / float(height))
        scales[image_id] = gain**2
    return scales


def _scaled_ground_truth_areas(coco_gt: COCO, area_scales: Mapping[Any, float], quiet: bool) -> COCO:
    """Copy a ground-truth COCO object and scale only bbox diagnostic areas."""

    dataset = copy.deepcopy(coco_gt.dataset)
    for annotation in dataset.get("annotations", []):
        image_id = annotation["image_id"]
        bbox = annotation.get("bbox")
        if bbox is None or len(bbox) != 4:
            raise ValueError("DIOR bbox area normalization requires four-value ground-truth bboxes")
        annotation["area"] = float(bbox[2]) * float(bbox[3]) * float(area_scales[image_id])
    scaled = COCO()
    scaled.dataset = dataset
    output_context = redirect_stdout(io.StringIO()) if quiet else nullcontext()
    with output_context:
        scaled.createIndex()
    return scaled


def build_detection_coco(
    coco_gt: COCO,
    predictions: Iterable[Mapping[str, Any]],
    quiet: bool = True,
    area_scales: Optional[Mapping[Any, float]] = None,
) -> COCO:
    """Build a COCO detections object, including the valid empty-result case."""

    valid_image_ids = set(coco_gt.getImgIds())
    annotations = []
    for index, source_annotation in enumerate(predictions, start=1):
        annotation = copy.deepcopy(dict(source_annotation))
        if annotation.get("image_id") not in valid_image_ids:
            raise ValueError(f"prediction references unknown image_id={annotation.get('image_id')!r}")
        bbox = annotation.get("bbox")
        if bbox is None or len(bbox) != 4:
            raise ValueError("bbox evaluation requires every prediction to contain [x, y, width, height]")
        if "category_id" not in annotation or "score" not in annotation:
            raise ValueError("every prediction must contain category_id and score")
        annotation["id"] = index
        area_scale = 1.0 if area_scales is None else float(area_scales[annotation["image_id"]])
        annotation["area"] = float(bbox[2]) * float(bbox[3]) * area_scale
        annotation["iscrowd"] = 0
        annotations.append(annotation)

    coco_dt = COCO()
    coco_dt.dataset = {
        "info": copy.deepcopy(coco_gt.dataset.get("info", {})),
        "images": copy.deepcopy(coco_gt.dataset.get("images", [])),
        "categories": copy.deepcopy(coco_gt.dataset.get("categories", [])),
        "annotations": annotations,
    }
    output_context = redirect_stdout(io.StringIO()) if quiet else nullcontext()
    with output_context:
        coco_dt.createIndex()
    return coco_dt


def evaluate_coco(
    ground_truth: CocoSource,
    predictions: PredictionSource,
    area_protocol: Union[str, AreaProtocol] = "dior",
    image_ids: Optional[Sequence[Any]] = None,
    category_ids: Optional[Sequence[Any]] = None,
    max_detections: Optional[int] = None,
    dior_input_size: Optional[int] = 640,
    quiet: bool = True,
) -> Dict[str, float]:
    """Evaluate bbox predictions with official COCO matching and paper ranges."""

    protocol = get_area_protocol(area_protocol)
    configured_max_detections = list(protocol.max_detections)
    if max_detections is not None:
        if int(max_detections) <= configured_max_detections[-2]:
            raise ValueError(f"max_detections must exceed the protocol diagnostic cap {configured_max_detections[-2]}")
        configured_max_detections[-1] = int(max_detections)
    coco_gt = load_coco_ground_truth(ground_truth, quiet=quiet)
    area_scales: Optional[Mapping[Any, float]] = None
    if protocol.name == "dior" and dior_input_size is not None:
        area_scales = _letterbox_area_scales(coco_gt, int(dior_input_size))
        coco_gt = _scaled_ground_truth_areas(coco_gt, area_scales, quiet=quiet)
    prediction_records = load_coco_predictions(predictions)
    coco_dt = build_detection_coco(coco_gt, prediction_records, quiet=quiet, area_scales=area_scales)
    evaluator = PaperCOCOeval(coco_gt, coco_dt, "bbox", protocol)
    evaluator.params.maxDets = configured_max_detections
    if image_ids is not None:
        evaluator.params.imgIds = list(image_ids)
    if category_ids is not None:
        evaluator.params.catIds = list(category_ids)

    output_context = redirect_stdout(io.StringIO()) if quiet else nullcontext()
    with output_context:
        evaluator.evaluate()
        evaluator.accumulate()
    metrics = evaluator.summary_metrics()
    evaluator.metrics = dict(metrics)
    evaluator.metric_names = tuple(metrics)
    evaluator.stats = np.asarray(tuple(metrics.values()), dtype=np.float64)
    return metrics


__all__ = [
    "AITOD_AREA_PROTOCOL",
    "COCO_MAX_AREA",
    "DIOR_AREA_PROTOCOL",
    "AreaProtocol",
    "PaperCOCOeval",
    "build_detection_coco",
    "configure_area_ranges",
    "evaluate_coco",
    "get_area_protocol",
    "load_coco_ground_truth",
    "load_coco_predictions",
]

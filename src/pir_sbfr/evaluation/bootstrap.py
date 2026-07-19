"""Image-paired bootstrap for contrasts between two COCO prediction sets."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .coco import (
    CocoSource,
    PredictionSource,
    evaluate_coco,
    get_area_protocol,
    load_coco_ground_truth,
    load_coco_predictions,
)


@dataclass(frozen=True)
class RemappedCocoSample:
    """One with-replacement image sample expanded to unique COCO image IDs."""

    ground_truth: Dict[str, Any]
    predictions_a: Tuple[Dict[str, Any], ...]
    predictions_b: Tuple[Dict[str, Any], ...]
    source_image_ids: Tuple[Any, ...]
    remapped_image_ids: Tuple[int, ...]

    @property
    def image_id_map(self) -> Tuple[Tuple[int, Any], ...]:
        """Pairs of ``(new_id, sampled_source_id)`` in sampling order."""

        return tuple(zip(self.remapped_image_ids, self.source_image_ids))


@dataclass(frozen=True)
class BootstrapResult:
    """Observed paired contrast and its bootstrap distribution.

    Deltas are always ``predictions_b - predictions_a`` and metric values are
    on COCO's native 0--1 scale.  Percentage-point forms are exposed by
    :meth:`to_dict` for direct comparison with the paper.
    """

    metric: str
    area_protocol: str
    replicates: int
    seed: int
    confidence: float
    score_a: float
    score_b: float
    observed_delta: float
    bootstrap_mean_delta: float
    bootstrap_standard_error: float
    confidence_interval: Tuple[float, float]
    two_sided_p_value: float
    bootstrap_deltas: Tuple[float, ...]

    def to_dict(self, include_distribution: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "metric": self.metric,
            "area_protocol": self.area_protocol,
            "replicates": self.replicates,
            "seed": self.seed,
            "confidence": self.confidence,
            "contrast": "predictions_b - predictions_a",
            "score_a": self.score_a,
            "score_b": self.score_b,
            "observed_delta": self.observed_delta,
            "observed_delta_percentage_points": 100.0 * self.observed_delta,
            "bootstrap_mean_delta": self.bootstrap_mean_delta,
            "bootstrap_mean_delta_percentage_points": 100.0 * self.bootstrap_mean_delta,
            "bootstrap_standard_error": self.bootstrap_standard_error,
            "confidence_interval": list(self.confidence_interval),
            "confidence_interval_percentage_points": [100.0 * value for value in self.confidence_interval],
            "two_sided_p_value": self.two_sided_p_value,
        }
        if include_distribution:
            result["bootstrap_deltas"] = list(self.bootstrap_deltas)
        return result


def _as_python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _index_by_image_id(records: Sequence[Mapping[str, Any]]) -> Dict[Any, list]:
    index: Dict[Any, list] = {}
    for record in records:
        if "image_id" not in record:
            raise ValueError("every annotation/prediction must contain image_id")
        index.setdefault(record["image_id"], []).append(record)
    return index


def remap_paired_coco_sample(
    ground_truth: Mapping[str, Any],
    predictions_a: Sequence[Mapping[str, Any]],
    predictions_b: Sequence[Mapping[str, Any]],
    sampled_image_ids: Sequence[Any],
) -> RemappedCocoSample:
    """Copy a with-replacement sample and give every occurrence a unique ID.

    Reusing an original ID for repeated draws would cause ``COCO.createIndex``
    to collapse image records and would silently discard bootstrap
    multiplicity.  This function expands images, annotations, and both models'
    detections in lockstep.
    """

    images_by_id: Dict[Any, Mapping[str, Any]] = {}
    for image in ground_truth.get("images", []):
        image_id = image.get("id")
        if image_id in images_by_id:
            raise ValueError(f"ground truth contains duplicate image id {image_id!r}")
        images_by_id[image_id] = image

    gt_by_image = _index_by_image_id(ground_truth.get("annotations", []))
    predictions_a_by_image = _index_by_image_id(predictions_a)
    predictions_b_by_image = _index_by_image_id(predictions_b)

    remapped_ground_truth = {
        key: copy.deepcopy(value)
        for key, value in ground_truth.items()
        if key not in {"images", "annotations"}
    }
    remapped_ground_truth.setdefault("info", {})
    remapped_ground_truth.setdefault("licenses", [])
    remapped_ground_truth.setdefault("categories", [])
    remapped_images = []
    remapped_annotations = []
    remapped_predictions_a = []
    remapped_predictions_b = []
    source_ids = []
    new_ids = []
    next_annotation_id = 1

    for occurrence, raw_source_id in enumerate(sampled_image_ids, start=1):
        source_id = _as_python_scalar(raw_source_id)
        if source_id not in images_by_id:
            raise KeyError(f"sampled image id {source_id!r} is absent from ground truth")
        new_image_id = occurrence
        source_ids.append(source_id)
        new_ids.append(new_image_id)

        image = copy.deepcopy(dict(images_by_id[source_id]))
        image["id"] = new_image_id
        remapped_images.append(image)

        for source_annotation in gt_by_image.get(source_id, ()):
            annotation = copy.deepcopy(dict(source_annotation))
            annotation["id"] = next_annotation_id
            annotation["image_id"] = new_image_id
            annotation.setdefault("iscrowd", 0)
            if "area" not in annotation:
                bbox = annotation.get("bbox")
                if bbox is None or len(bbox) != 4:
                    raise ValueError("ground-truth annotations need area or a four-value bbox")
                annotation["area"] = float(bbox[2]) * float(bbox[3])
            remapped_annotations.append(annotation)
            next_annotation_id += 1

        for source_prediction in predictions_a_by_image.get(source_id, ()):
            prediction = copy.deepcopy(dict(source_prediction))
            prediction.pop("id", None)
            prediction["image_id"] = new_image_id
            remapped_predictions_a.append(prediction)

        for source_prediction in predictions_b_by_image.get(source_id, ()):
            prediction = copy.deepcopy(dict(source_prediction))
            prediction.pop("id", None)
            prediction["image_id"] = new_image_id
            remapped_predictions_b.append(prediction)

    remapped_ground_truth["images"] = remapped_images
    remapped_ground_truth["annotations"] = remapped_annotations
    return RemappedCocoSample(
        ground_truth=remapped_ground_truth,
        predictions_a=tuple(remapped_predictions_a),
        predictions_b=tuple(remapped_predictions_b),
        source_image_ids=tuple(source_ids),
        remapped_image_ids=tuple(new_ids),
    )


ScoreFunction = Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], float]
ProgressFunction = Callable[[int, int], None]


def _dataset_mapping(source: CocoSource) -> Dict[str, Any]:
    coco = load_coco_ground_truth(source, quiet=True)
    return copy.deepcopy(coco.dataset)


def _checked_score(
    score_function: ScoreFunction,
    ground_truth: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
) -> float:
    value = float(score_function(ground_truth, predictions))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"evaluation returned an invalid score: {value!r}")
    return value


def paired_bootstrap_coco(
    ground_truth: CocoSource,
    predictions_a: PredictionSource,
    predictions_b: PredictionSource,
    area_protocol: str = "dior",
    metric: str = "AP",
    replicates: int = 10_000,
    seed: int = 20_260_718,
    confidence: float = 0.95,
    max_detections: Optional[int] = None,
    dior_input_size: Optional[int] = 640,
    score_function: Optional[ScoreFunction] = None,
    progress: Optional[ProgressFunction] = None,
) -> BootstrapResult:
    """Recompute paired COCO AP for image-level bootstrap replicates.

    The same sampled image IDs are used for both prediction sets.  Sampling is
    with replacement to the original test-set size, and every repeated draw is
    expanded by :func:`remap_paired_coco_sample` before evaluation.
    """

    if int(replicates) < 1:
        raise ValueError("replicates must be at least one")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    protocol = get_area_protocol(area_protocol)
    dataset = _dataset_mapping(ground_truth)
    records_a = tuple(load_coco_predictions(predictions_a))
    records_b = tuple(load_coco_predictions(predictions_b))
    image_ids = tuple(image["id"] for image in dataset.get("images", []))
    if not image_ids:
        raise ValueError("ground truth contains no images")

    if score_function is None:

        def paper_score(gt: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]) -> float:
            metrics = evaluate_coco(
                gt,
                predictions,
                area_protocol=protocol,
                max_detections=max_detections,
                dior_input_size=dior_input_size,
                quiet=True,
            )
            if metric not in metrics:
                choices = ", ".join(metrics)
                raise KeyError(f"metric {metric!r} is unavailable; choose one of: {choices}")
            return float(metrics[metric])

        active_score_function = paper_score
    else:
        active_score_function = score_function

    score_a = _checked_score(active_score_function, dataset, records_a)
    score_b = _checked_score(active_score_function, dataset, records_b)
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(replicates), dtype=np.float64)
    number_of_images = len(image_ids)

    for replicate_index in range(int(replicates)):
        sampled_indices = rng.integers(0, number_of_images, size=number_of_images)
        sampled_ids = tuple(image_ids[int(index)] for index in sampled_indices)
        remapped = remap_paired_coco_sample(dataset, records_a, records_b, sampled_ids)
        replicate_a = _checked_score(active_score_function, remapped.ground_truth, remapped.predictions_a)
        replicate_b = _checked_score(active_score_function, remapped.ground_truth, remapped.predictions_b)
        deltas[replicate_index] = replicate_b - replicate_a
        if progress is not None:
            progress(replicate_index + 1, int(replicates))

    alpha = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(deltas, [alpha, 1.0 - alpha])
    non_positive = int(np.count_nonzero(deltas <= 0.0))
    non_negative = int(np.count_nonzero(deltas >= 0.0))
    p_value = min(1.0, 2.0 * (min(non_positive, non_negative) + 1.0) / (len(deltas) + 1.0))
    standard_error = float(np.std(deltas, ddof=1 if len(deltas) > 1 else 0))
    observed_delta = score_b - score_a
    return BootstrapResult(
        metric=str(metric),
        area_protocol=protocol.name,
        replicates=int(replicates),
        seed=int(seed),
        confidence=float(confidence),
        score_a=score_a,
        score_b=score_b,
        observed_delta=observed_delta,
        bootstrap_mean_delta=float(np.mean(deltas)),
        bootstrap_standard_error=standard_error,
        confidence_interval=(float(lower), float(upper)),
        two_sided_p_value=float(p_value),
        bootstrap_deltas=tuple(float(value) for value in deltas),
    )


__all__ = [
    "BootstrapResult",
    "RemappedCocoSample",
    "paired_bootstrap_coco",
    "remap_paired_coco_sample",
]

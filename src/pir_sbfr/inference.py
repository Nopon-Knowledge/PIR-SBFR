"""Checkpoint loading, image preprocessing and metadata-aware inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor

from ultralytics.utils.ops import non_max_suppression

from pir_sbfr.data.degradations import REFERENCE_METADATA
from pir_sbfr.models.detector import PIRSBFRModel


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass
class LetterboxInfo:
    ratio: float
    pad_x: float
    pad_y: float
    original_hw: Tuple[int, int]


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device if device != "cuda" else "cuda:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint_model(weights: Union[str, Path], device: Optional[str] = None) -> PIRSBFRModel:
    """Load an Ultralytics trainer checkpoint with explicit trusted-file semantics."""
    path = Path(weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    target = resolve_device(device)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only keyword
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, PIRSBFRModel):
        model = checkpoint
    elif isinstance(checkpoint, Mapping):
        model = checkpoint.get("ema") or checkpoint.get("model")
    else:
        model = None
    if not isinstance(model, PIRSBFRModel):
        raise TypeError(f"{path} does not contain a PIRSBFRModel checkpoint")
    model = model.float().to(target).eval()
    return model


def letterbox_rgb(image_rgb: np.ndarray, size: int = 640) -> Tuple[np.ndarray, LetterboxInfo]:
    """Resize with preserved aspect ratio and symmetric YOLO padding."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("expected HWC RGB image")
    if np.issubdtype(image_rgb.dtype, np.floating):
        if not np.isfinite(image_rgb).all() or float(image_rgb.min()) < 0.0 or float(image_rgb.max()) > 1.0:
            raise ValueError("floating RGB images must be finite and normalized to [0,1]")
        padding_value = (114.0 / 255.0,) * 3
    elif np.issubdtype(image_rgb.dtype, np.integer):
        maximum = float(np.iinfo(image_rgb.dtype).max)
        padding_value = (round(114.0 * maximum / 255.0),) * 3
    else:
        raise TypeError(f"unsupported RGB dtype: {image_rgb.dtype}")
    original_h, original_w = image_rgb.shape[:2]
    ratio = min(float(size) / original_h, float(size) / original_w)
    resized_w = max(1, int(round(original_w * ratio)))
    resized_h = max(1, int(round(original_h * ratio)))
    resized = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_x_total = size - resized_w
    pad_y_total = size - resized_h
    left = int(round(pad_x_total / 2.0 - 0.1))
    right = pad_x_total - left
    top = int(round(pad_y_total / 2.0 - 0.1))
    bottom = pad_y_total - top
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=padding_value,
    )
    return padded, LetterboxInfo(ratio, float(left), float(top), (original_h, original_w))


def image_tensor_rgb(image_rgb: np.ndarray, device: torch.device) -> Tensor:
    """Convert integer RGB or normalized float RGB to one float CHW tensor."""
    if np.issubdtype(image_rgb.dtype, np.floating) and (
        not np.isfinite(image_rgb).all() or float(image_rgb.min()) < 0.0 or float(image_rgb.max()) > 1.0
    ):
        raise ValueError("floating RGB images must be finite and normalized to [0,1]")
    array = np.ascontiguousarray(image_rgb.transpose(2, 0, 1))
    tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32)
    if np.issubdtype(image_rgb.dtype, np.integer):
        tensor = tensor / float(np.iinfo(image_rgb.dtype).max)
    elif not np.issubdtype(image_rgb.dtype, np.floating):
        raise TypeError(f"unsupported RGB dtype: {image_rgb.dtype}")
    return tensor


def restore_boxes(boxes: Tensor, info: LetterboxInfo) -> Tensor:
    """Map input-space xyxy boxes back to the original image."""
    restored = boxes.clone()
    restored[:, [0, 2]] = (restored[:, [0, 2]] - info.pad_x) / info.ratio
    restored[:, [1, 3]] = (restored[:, [1, 3]] - info.pad_y) / info.ratio
    original_h, original_w = info.original_hw
    restored[:, [0, 2]] = restored[:, [0, 2]].clamp(0, original_w)
    restored[:, [1, 3]] = restored[:, [1, 3]].clamp(0, original_h)
    return restored


def acquisition_tensors(
    batch_size: int,
    device: torch.device,
    metadata: Optional[Sequence[float]] = None,
    availability: Optional[Sequence[float]] = None,
) -> Tuple[Tensor, Tensor]:
    values_source = REFERENCE_METADATA if metadata is None else metadata
    if availability is None:
        mask_source = (0.0, 0.0, 0.0) if metadata is None else (1.0, 1.0, 1.0)
    else:
        mask_source = availability
    values = tuple(float(v) for v in values_source)
    mask = tuple(float(v) for v in mask_source)
    if len(values) != 3 or len(mask) != 3:
        raise ValueError("metadata and availability must each contain three values")
    return (
        torch.tensor(values, device=device).view(1, 3).expand(batch_size, -1),
        torch.tensor(mask, device=device).view(1, 3).expand(batch_size, -1),
    )


def infer_image(
    model: PIRSBFRModel,
    image_rgb: np.ndarray,
    metadata: Optional[Sequence[float]] = None,
    availability: Optional[Sequence[float]] = None,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    max_det: int = 300,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Return ``[x1,y1,x2,y2,confidence,class]`` and router diagnostics."""
    prepared, info = letterbox_rgb(image_rgb, imgsz)
    image_tensor = image_tensor_rgb(prepared, next(model.parameters()).device).unsqueeze(0)
    values, mask = acquisition_tensors(1, image_tensor.device, metadata, availability)
    with torch.inference_mode():
        output, aux = model.predict(image_tensor, metadata=values, availability=mask, return_aux=True)
        decoded = output[0] if isinstance(output, tuple) else output
        detections = non_max_suppression(decoded, conf_thres=conf, iou_thres=iou, max_det=max_det)[0]
    # Tensors created in inference_mode reject later in-place edits; clone into
    # a regular tensor before undoing letterbox coordinates.
    detections = detections.detach().clone()
    if detections.numel():
        detections[:, :4] = restore_boxes(detections[:, :4], info)
    return detections.detach().cpu(), {key: value.detach().cpu() for key, value in aux.items()}


def infer_images(
    model: PIRSBFRModel,
    images_rgb: Sequence[np.ndarray],
    metadata: Sequence[Sequence[float]],
    availability: Sequence[Sequence[float]],
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    max_det: int = 300,
) -> Tuple[List[Tensor], Dict[str, Tensor]]:
    """Batched variant of :func:`infer_image` for benchmark-scale evaluation."""
    if not images_rgb:
        return [], {}
    if len(images_rgb) != len(metadata) or len(images_rgb) != len(availability):
        raise ValueError("images, metadata and availability must have the same length")
    prepared: List[Tensor] = []
    infos: List[LetterboxInfo] = []
    device = next(model.parameters()).device
    for image in images_rgb:
        padded, info = letterbox_rgb(image, imgsz)
        prepared.append(image_tensor_rgb(padded, device))
        infos.append(info)
    batch = torch.stack(prepared)
    values = torch.tensor(metadata, device=device, dtype=batch.dtype)
    masks = torch.tensor(availability, device=device, dtype=batch.dtype)
    with torch.inference_mode():
        output, aux = model.predict(batch, metadata=values, availability=masks, return_aux=True)
        decoded = output[0] if isinstance(output, tuple) else output
        detections = non_max_suppression(decoded, conf_thres=conf, iou_thres=iou, max_det=max_det)
    restored: List[Tensor] = []
    for detection, info in zip(detections, infos):
        detection = detection.detach().clone()
        if detection.numel():
            detection[:, :4] = restore_boxes(detection[:, :4], info)
        restored.append(detection.cpu())
    return restored, {key: value.detach().cpu() for key, value in aux.items()}


def image_paths(source: Union[str, Path]) -> List[Path]:
    path = Path(source).expanduser()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image suffix: {path.suffix}")
        return [path.resolve()]
    if path.is_dir():
        return sorted(p.resolve() for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(path)


def load_metadata_records(path: Optional[Union[str, Path]]) -> Dict[str, Dict]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, dict):
        raise ValueError("metadata JSON must map file names/paths to descriptor objects")
    return records


def record_acquisition(record: Optional[Mapping]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if not record:
        return REFERENCE_METADATA, (0.0, 0.0, 0.0)
    values = (
        float(record.get("gsd", REFERENCE_METADATA[0])),
        float(record.get("mtf", REFERENCE_METADATA[1])),
        float(record.get("snr", REFERENCE_METADATA[2])),
    )
    if "availability" in record:
        mask = tuple(float(v) for v in record["availability"])
    else:
        mask = tuple(float(key in record) for key in ("gsd", "mtf", "snr"))
    return values, mask


def find_metadata_record(records: Mapping[str, Dict], path: Path) -> Optional[Dict]:
    for key in (str(path), path.name, path.stem):
        if key in records:
            return records[key]
    return None

"""Detection, scale-supervision and paired degradation-consistency objectives."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


def object_scale_distribution(
    batch: Dict[str, Tensor],
    batch_size: int,
    image_hw: Sequence[int],
    mode: str = "dior",
    eps: float = 1e-6,
) -> Tensor:
    """Build the annotation-only scene scale target from paper equation (15).

    Ultralytics stores normalized ``xywh`` boxes after all geometric/mosaic
    transforms, so the counts here describe final visible boxes at the actual
    network input size, as required by the paper.
    """
    device = batch["bboxes"].device
    dtype = batch["bboxes"].dtype
    counts = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    if batch["bboxes"].numel() == 0:
        return torch.full_like(counts, 1.0 / 3.0)

    height, width = float(image_hw[0]), float(image_hw[1])
    boxes = batch["bboxes"]
    box_batch = batch["batch_idx"].view(-1).long()
    pixel_w = boxes[:, 2] * width
    pixel_h = boxes[:, 3] * height

    normalized_mode = mode.lower().replace("-", "").replace("_", "")
    if normalized_mode == "dior":
        area = pixel_w * pixel_h
        groups = torch.where(area < 32.0**2, 0, torch.where(area < 96.0**2, 1, 2))
    elif normalized_mode in {"aitod", "aitodv2"}:
        # AI-TOD defines absolute size as sqrt(area). VT+T (2-16 px) route
        # to P3, S (16-32) to P4, and M (32-64) to P5. The public set is
        # designed around 2-64 px objects; outliers are clamped to edge bins.
        absolute_size = (pixel_w * pixel_h).clamp_min(0.0).sqrt()
        groups = torch.where(absolute_size < 16.0, 0, torch.where(absolute_size < 32.0, 1, 2))
    else:
        raise ValueError("scale mode must be 'dior' or 'aitodv2'")

    flat_index = box_batch * 3 + groups.long()
    flat_counts = torch.bincount(flat_index, minlength=batch_size * 3).to(dtype=dtype)
    counts = flat_counts.view(batch_size, 3)
    return (counts + eps) / (counts.sum(dim=1, keepdim=True) + 3.0 * eps)


def scale_kl_divergence(target: Tensor, estimate: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute ``D_KL(target || estimate)`` averaged across images."""
    target = target.clamp_min(eps)
    estimate = estimate.clamp_min(eps)
    return (target * (target.log() - estimate.log())).sum(dim=-1).mean()


class PIRSBFRLoss:
    """Paper equation (22) on top of Ultralytics' native YOLO11 detection loss."""

    def __init__(
        self,
        model,
        lambda_scale: float = 0.1,
        lambda_consistency: float = 0.1,
        scale_mode: str = "dior",
        eps: float = 1e-6,
    ) -> None:
        self.det = v8DetectionLoss(model)
        self.lambda_scale = float(lambda_scale)
        self.lambda_consistency = float(lambda_consistency)
        self.scale_mode = scale_mode
        self.eps = float(eps)

    def _dense_predictions(self, feats: Sequence[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Decode raw YOLO11 feature-map predictions for consistency matching."""
        no = self.det.no
        batch_size = feats[0].shape[0]
        pred_distri, pred_scores = torch.cat([x.view(batch_size, no, -1) for x in feats], dim=2).split(
            (self.det.reg_max * 4, self.det.nc), dim=1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(feats, self.det.stride, 0.5)
        pred_boxes_grid = self.det.bbox_decode(anchor_points, pred_distri)
        return pred_distri, pred_scores, pred_boxes_grid, anchor_points, stride_tensor

    def _positive_mask(self, feats: Sequence[Tensor], batch: Dict[str, Tensor]) -> Tensor:
        """Use clean-view TaskAligned assignments as the shared set R in equation (21)."""
        _, pred_scores, pred_boxes, anchor_points, stride_tensor = self._dense_predictions(feats)
        batch_size = pred_scores.shape[0]
        dtype = pred_scores.dtype
        image_size = torch.tensor(feats[0].shape[2:], device=pred_scores.device, dtype=dtype) * self.det.stride[0]
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), dim=1
        )
        targets = self.det.preprocess(
            targets.to(self.det.device), batch_size, scale_tensor=image_size[[1, 0, 1, 0]]
        )
        gt_labels, gt_boxes = targets.split((1, 4), dim=2)
        mask_gt = gt_boxes.sum(dim=2, keepdim=True).gt_(0.0)
        _, _, _, foreground, _ = self.det.assigner(
            pred_scores.detach().sigmoid(),
            (pred_boxes.detach() * stride_tensor).type(gt_boxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_boxes,
            mask_gt,
        )
        return foreground

    def consistency_loss(
        self,
        clean_feats: Sequence[Tensor],
        degraded_feats: Sequence[Tensor],
        batch: Dict[str, Tensor],
    ) -> Tensor:
        """Symmetric class KL plus normalized-box L1 on clean assignments."""
        _, clean_scores, clean_boxes, _, clean_stride = self._dense_predictions(clean_feats)
        _, degraded_scores, degraded_boxes, _, degraded_stride = self._dense_predictions(degraded_feats)
        if clean_scores.shape != degraded_scores.shape:
            raise ValueError("paired views must yield identical dense prediction shapes")

        foreground = self._positive_mask(clean_feats, batch)
        if not bool(foreground.any()):
            return (clean_scores.sum() + degraded_scores.sum()) * 0.0

        p_log = F.log_softmax(clean_scores[foreground], dim=-1)
        q_log = F.log_softmax(degraded_scores[foreground], dim=-1)
        p = p_log.exp()
        q = q_log.exp()
        symmetric_kl = 0.5 * (
            (p * (p_log - q_log)).sum(dim=-1) + (q * (q_log - p_log)).sum(dim=-1)
        )

        image_h, image_w = batch["img"].shape[-2:]
        normalizer = clean_boxes.new_tensor([image_w, image_h, image_w, image_h]).view(1, 1, 4)
        clean_normalized = clean_boxes * clean_stride / normalizer
        degraded_normalized = degraded_boxes * degraded_stride / normalizer
        box_l1 = (clean_normalized[foreground] - degraded_normalized[foreground]).abs().sum(dim=-1)
        return (symmetric_kl + box_l1).mean()

    def __call__(
        self,
        clean_preds: Sequence[Tensor],
        clean_aux: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        degraded_preds: Optional[Sequence[Tensor]] = None,
        degraded_aux: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = int(batch["img"].shape[0])
        clean_det, clean_items = self.det(clean_preds, batch)

        if degraded_preds is None:
            scale_value = clean_det.new_zeros(())
            total = clean_det
            if "scale_estimate" in clean_aux and self.lambda_scale != 0.0:
                target_scale = object_scale_distribution(
                    batch,
                    batch_size=batch_size,
                    image_hw=batch["img"].shape[-2:],
                    mode=self.scale_mode,
                    eps=self.eps,
                ).to(device=clean_aux["scale_estimate"].device, dtype=clean_aux["scale_estimate"].dtype)
                scale_value = self.lambda_scale * scale_kl_divergence(
                    target_scale, clean_aux["scale_estimate"], self.eps
                )
                total = total + batch_size * scale_value
            return total, torch.cat(
                (clean_items, scale_value.detach().view(1), clean_det.detach().new_zeros(1))
            )

        degraded_det, degraded_items = self.det(degraded_preds, batch)
        detection = 0.5 * (clean_det + degraded_det)
        detection_items = 0.5 * (clean_items + degraded_items)

        scale_loss = detection.new_zeros(())
        if self.lambda_scale != 0.0:
            target_scale = object_scale_distribution(
                batch,
                batch_size=batch_size,
                image_hw=batch["img"].shape[-2:],
                mode=self.scale_mode,
                eps=self.eps,
            )
            target_scale = target_scale.to(
                device=clean_aux["scale_estimate"].device,
                dtype=clean_aux["scale_estimate"].dtype,
            )
            scale_loss = scale_kl_divergence(target_scale, clean_aux["scale_estimate"], self.eps)
            if degraded_aux is not None:
                scale_loss = 0.5 * (
                    scale_loss + scale_kl_divergence(target_scale, degraded_aux["scale_estimate"], self.eps)
                )
        consistency = detection.new_zeros(())
        if self.lambda_consistency != 0.0:
            consistency = self.consistency_loss(clean_preds, degraded_preds, batch)

        weighted_scale = self.lambda_scale * scale_loss
        weighted_consistency = self.lambda_consistency * consistency
        total = detection + batch_size * (weighted_scale + weighted_consistency)
        items = torch.cat(
            (
                detection_items,
                weighted_scale.detach().view(1),
                weighted_consistency.detach().view(1),
            )
        )
        return total, items

"""End-to-end PIR-SBFR detector using YOLO11's native Detect head."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils.torch_utils import initialize_weights

from .blocks import FACH, YOLO11DRFBBackbone
from .loss import PIRSBFRLoss
from .router import PIRSBFRNeck


class PIRSBFRModel(BaseModel):
    """Runnable official implementation model.

    The last element of ``self.model`` is the official Ultralytics ``Detect``
    module. This preserves YOLO11 decoding, DFL, TaskAligned assignment and
    validator compatibility while all paper-specific components remain local.
    """

    def __init__(
        self,
        nc: int = 20,
        scale_mode: str = "dior",
        router_config: Optional[Dict] = None,
        fach_experts: int = 3,
        lambda_scale: float = 0.1,
        lambda_consistency: float = 0.1,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        router_config = dict(router_config or {})
        backbone = YOLO11DRFBBackbone()
        neck = PIRSBFRNeck(in_channels=backbone.out_channels, **router_config)
        fach = FACH(neck.fpn_pan.out_channels, num_experts=fach_experts)
        detect = Detect(nc=int(nc), ch=neck.fpn_pan.out_channels)
        self.model = nn.ModuleList((backbone, neck, fach, detect))
        self.save = []
        self.yaml = {
            "nc": int(nc),
            "scale_mode": scale_mode,
            "router": router_config,
            "fach_experts": int(fach_experts),
        }
        self.names = {index: str(index) for index in range(int(nc))}
        self.nc = int(nc)
        self.inplace = True
        self.end2end = False
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        detect.inplace = True
        detect.stride = self.stride.clone()

        # Trainer replaces this namespace with its full configuration. Keeping
        # native YOLO11 gains here makes standalone unit/smoke tests valid.
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        self.scale_mode = scale_mode
        self.lambda_scale = float(lambda_scale)
        self.lambda_consistency = float(lambda_consistency)
        self.last_routing: Dict[str, Tensor] = {}
        self.criterion = None

        initialize_weights(self)
        detect.bias_init()
        if verbose:
            self.info(imgsz=640)

    @property
    def backbone(self) -> YOLO11DRFBBackbone:
        return self.model[0]

    @property
    def neck(self) -> PIRSBFRNeck:
        return self.model[1]

    @property
    def fach(self) -> FACH:
        return self.model[2]

    @property
    def detect(self) -> Detect:
        return self.model[3]

    def _forward_detector(
        self,
        images: Tensor,
        metadata: Optional[Tensor] = None,
        availability: Optional[Tensor] = None,
    ) -> Tuple[object, Dict[str, Tensor]]:
        pyramid = self.backbone(images)
        neck_features, aux = self.neck(pyramid, metadata, availability, return_aux=True)
        coupled = self.fach(neck_features)
        predictions = self.detect(list(coupled))
        self.last_routing = {key: value.detach() for key, value in aux.items()}
        return predictions, aux

    def predict(
        self,
        x: Tensor,
        profile: bool = False,
        visualize: bool = False,
        augment: bool = False,
        embed=None,
        metadata: Optional[Tensor] = None,
        availability: Optional[Tensor] = None,
        return_aux: bool = False,
    ):
        """Inference with optional per-image ``[gsd, mtf, snr]`` metadata."""
        if augment:
            raise NotImplementedError("test-time augmentation is not defined for metadata-conditioned inference")
        del profile, visualize, embed
        predictions, aux = self._forward_detector(x, metadata, availability)
        return (predictions, aux) if return_aux else predictions

    def loss(self, batch: Dict[str, Tensor], preds=None):
        """Compute clean-only validation loss or the complete paired training loss."""
        if self.criterion is None:
            self.criterion = self.init_criterion()

        if preds is None:
            clean_preds, clean_aux = self._forward_detector(
                batch["img"], batch.get("metadata"), batch.get("availability")
            )
        else:
            # During validation Ultralytics passes Detect's inference tuple
            # ``(decoded, raw_feature_maps)``. Scale auxiliaries are unnecessary
            # for the clean-only loss path, so retain the tuple for native loss.
            if (
                isinstance(preds, tuple)
                and len(preds) == 2
                and isinstance(preds[0], Tensor)
                and isinstance(preds[1], (list, tuple))
            ):
                clean_preds, clean_aux = preds, {}
            else:
                clean_preds, clean_aux = preds

        degraded_preds = None
        degraded_aux = None
        if "img_degraded" in batch:
            degraded_preds, degraded_aux = self._forward_detector(
                batch["img_degraded"],
                batch.get("metadata_degraded"),
                batch.get("availability_degraded", batch.get("availability")),
            )
        return self.criterion(clean_preds, clean_aux, batch, degraded_preds, degraded_aux)

    def init_criterion(self) -> PIRSBFRLoss:
        return PIRSBFRLoss(
            self,
            lambda_scale=self.lambda_scale,
            lambda_consistency=self.lambda_consistency,
            scale_mode=self.scale_mode,
        )

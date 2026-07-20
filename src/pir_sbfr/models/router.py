"""PIR-SBFR routing neck from paper equations (12)-(20)."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv

from .blocks import FPNPAN
from .physical import PhysicalReliabilityPrior


class ResidualExpert(nn.Module):
    """Bounded visual-routing expert ``T_k``.

    The paper leaves ``T_k`` unspecified while describing the correction as
    bounded. We therefore use a two-layer MLP and explicit tanh bound. This is
    one of the documented implementation choices, not a claimed hidden detail.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 3, bound: float = 4.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.bound = float(bound)

    def forward(self, z: Tensor) -> Tensor:
        return torch.tanh(self.net(z)) * self.bound


class VisualResidualRouter(nn.Module):
    """K-branch scale-supervised visual residual from equations (12)-(14)."""

    def __init__(
        self,
        aligned_channels: int = 64,
        num_experts: int = 4,
        hidden_dim: int = 96,
        residual_bound: float = 4.0,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        input_dim = aligned_channels * 3
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList(
            ResidualExpert(input_dim, hidden_dim, 3, residual_bound) for _ in range(num_experts)
        )

    def forward(self, pyramid: Sequence[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        if len(pyramid) != 3:
            raise ValueError("visual router expects P3, P4 and P5")
        z = torch.cat([F.adaptive_avg_pool2d(p, 1).flatten(1) for p in pyramid], dim=1)
        mixture_weights = torch.softmax(self.gate(z), dim=-1)
        residual = torch.zeros(z.shape[0], 3, device=z.device, dtype=z.dtype)
        for index, expert in enumerate(self.experts):
            residual = residual + mixture_weights[:, index : index + 1] * expert(z)
        scale_estimate = torch.softmax(residual, dim=-1)
        return residual, scale_estimate, mixture_weights


class PIRSBFRNeck(nn.Module):
    """Physical Imaging Reliability-Guided Scale-Biased Feature Reweighting.

    Args:
        in_channels: Backbone P3/P4/P5 channel counts.
        aligned_channels: Common channel count for ``A_i(P_i)``.
        eta: Physical-prior logit strength from equation (17).
        temperature: Routing temperature ``tau``.
        physical_fields: Optional length-three mask for GSD/MTF/SNR ablations.
        use_physical: Disable for the internal visual-only baseline.
        use_visual: Disable for analytic-only ablations.
        p5_bypass: Enable the unweighted coarse structural path in equation (20).
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (128, 128, 256),
        aligned_channels: int = 64,
        num_experts: int = 4,
        visual_hidden_dim: int = 280,
        residual_bound: float = 4.0,
        eta: float = 1.0,
        temperature: float = 1.0,
        eps: float = 1e-6,
        physical_fields: Sequence[float] = (1.0, 1.0, 1.0),
        use_physical: bool = True,
        use_visual: bool = True,
        p5_bypass: bool = True,
    ) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("in_channels must describe P3, P4 and P5")
        if len(physical_fields) != 3:
            raise ValueError("physical_fields must contain GSD, MTF and SNR switches")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.align = nn.ModuleList(Conv(int(c), aligned_channels, 1, 1) for c in in_channels)
        self.visual = (
            VisualResidualRouter(
                aligned_channels=aligned_channels,
                num_experts=num_experts,
                hidden_dim=visual_hidden_dim,
                residual_bound=residual_bound,
            )
            if use_visual
            else None
        )
        self.physical = PhysicalReliabilityPrior(eps=eps)
        self.fpn_pan = FPNPAN(aligned_channels)
        # Bypass-disabled ablations must not register dead bypass parameters.
        self.p5_projection = Conv(int(in_channels[-1]), 256, 1, 1) if p5_bypass else None
        self.register_buffer("physical_fields", torch.tensor(physical_fields, dtype=torch.float32), persistent=True)
        self.eta = float(eta)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.use_physical = bool(use_physical)
        self.use_visual = bool(use_visual)
        self.p5_bypass = bool(p5_bypass)

    def _default_acquisition(self, feature: Tensor) -> Tuple[Tensor, Tensor]:
        batch = feature.shape[0]
        ref = self.physical.reference.to(device=feature.device, dtype=feature.dtype).view(1, 3)
        return ref.expand(batch, -1), torch.zeros(batch, 3, device=feature.device, dtype=feature.dtype)

    def forward(
        self,
        pyramid: Sequence[Tensor],
        metadata: Optional[Tensor] = None,
        availability: Optional[Tensor] = None,
        return_aux: bool = True,
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor], Dict[str, Tensor]]:
        if len(pyramid) != 3:
            raise ValueError("PIR-SBFR requires P3, P4 and P5")
        aligned = [projection(feature) for projection, feature in zip(self.align, pyramid)]
        batch = aligned[0].shape[0]

        if self.use_visual:
            if self.visual is None:
                raise RuntimeError("visual router was disabled at construction")
            delta_vis, scale_estimate, expert_weights = self.visual(aligned)
        else:
            delta_vis = torch.zeros(batch, 3, device=aligned[0].device, dtype=aligned[0].dtype)
            scale_estimate = torch.full_like(delta_vis, 1.0 / 3.0)
            expert_weights = torch.empty(batch, 0, device=aligned[0].device, dtype=aligned[0].dtype)

        if metadata is None:
            metadata, default_mask = self._default_acquisition(aligned[0])
            if availability is None:
                availability = default_mask
        if availability is None:
            availability = torch.ones_like(metadata)
        if metadata.shape != (batch, 3) or availability.shape != (batch, 3):
            raise ValueError("metadata and availability must both have shape [batch, 3]")

        field_mask = self.physical_fields.to(device=availability.device, dtype=availability.dtype).view(1, 3)
        effective_mask = availability * field_mask
        if self.use_physical:
            rho_phy = self.physical(metadata, effective_mask)
            physical_logits = self.eta * torch.log(rho_phy.clamp_min(self.eps))
        else:
            rho_phy = torch.ones(batch, 3, device=aligned[0].device, dtype=aligned[0].dtype)
            physical_logits = torch.zeros_like(rho_phy)

        routing_logits = (delta_vis + physical_logits) / self.temperature
        routing_weights = torch.softmax(routing_logits, dim=-1)
        reweighted = [
            3.0 * routing_weights[:, i].view(-1, 1, 1, 1) * feature for i, feature in enumerate(aligned)
        ]
        f3, f4, f5 = self.fpn_pan(reweighted)
        if self.p5_bypass:
            if self.p5_projection is None:
                raise RuntimeError("P5 bypass was enabled without a projection")
            f5 = f5 + self.p5_projection(pyramid[-1])

        aux = {
            "weights": routing_weights,
            "rho_phy": rho_phy,
            "delta_vis": delta_vis,
            "scale_estimate": scale_estimate,
            "expert_weights": expert_weights,
            "degradation": self.physical.degradation_coordinates(metadata, effective_mask),
            "metadata": metadata,
            "availability": effective_mask,
        }
        return (f3, f4, f5), aux if return_aux else {}

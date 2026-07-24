"""Image-formation-informed analytic reliability prior from manuscript equations (7), (9)-(11)."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn


class PhysicalReliabilityPrior(nn.Module):
    """Map GSD, MTF/PSF sharpness, and SNR to P3-P5 analytic reliability.

    This calibration-free monotonic module deliberately has no trainable parameters.
    Missing descriptor values are
    replaced by their reference values *before* nonlinear transforms are evaluated,
    then removed with the availability mask. This prevents invalid placeholders (for
    example, a missing GSD encoded as zero) from generating NaNs before masking.
    """

    def __init__(
        self,
        reference: Sequence[float] = (1.0, 0.5, 30.0),
        strides: Sequence[int] = (8, 16, 32),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(reference) != 3:
            raise ValueError("reference must contain [gsd, mtf, snr]")
        if len(strides) != 3:
            raise ValueError("PIR-SBFR requires exactly three strides (P3, P4, P5)")
        if any(s <= 0 for s in strides):
            raise ValueError("strides must be positive")

        ref = torch.tensor(reference, dtype=torch.float32)
        kappa = torch.tensor([[8.0 / float(s)] * 3 for s in strides], dtype=torch.float32)
        self.register_buffer("reference", ref, persistent=True)
        self.register_buffer("kappa", kappa, persistent=True)
        self.eps = float(eps)

    def degradation_coordinates(self, metadata: Tensor, availability: Optional[Tensor] = None) -> Tensor:
        """Return non-negative ``[d_g, d_q, d_s]`` coordinates from equation (9)."""
        if metadata.ndim != 2 or metadata.shape[-1] != 3:
            raise ValueError("metadata must have shape [batch, 3]")
        if availability is None:
            availability = torch.ones_like(metadata)
        if availability.shape != metadata.shape:
            raise ValueError("availability must have the same shape as metadata")

        mask = availability.to(dtype=metadata.dtype).clamp_(0.0, 1.0)
        ref = self.reference.to(device=metadata.device, dtype=metadata.dtype).view(1, 3)
        safe = torch.where(mask > 0, metadata, ref)

        gsd = safe[:, 0].clamp_min(self.eps)
        mtf = safe[:, 1]
        snr = safe[:, 2]
        dg = torch.log(gsd / ref[:, 0]).clamp_min(0.0)
        dq = (1.0 - mtf / ref[:, 1]).clamp_min(0.0)
        ds = ((ref[:, 2] - snr) / ref[:, 2]).clamp_min(0.0)
        return torch.stack((dg, dq, ds), dim=-1) * mask

    def forward(self, metadata: Tensor, availability: Optional[Tensor] = None) -> Tensor:
        """Return analytic reliabilities ``rho_phy`` with shape ``[batch, 3]``."""
        degradation = self.degradation_coordinates(metadata, availability)
        kappa = self.kappa.to(device=metadata.device, dtype=metadata.dtype)
        return torch.exp(-degradation @ kappa.transpose(0, 1))

    def extra_repr(self) -> str:
        ref = tuple(float(x) for x in self.reference.tolist())
        return f"reference={ref}, eps={self.eps:g}, trainable=False"

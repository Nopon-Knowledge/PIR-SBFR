"""DRFB backbone and FACH head blocks described in paper equations (3)-(6), (22)-(24)."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ultralytics.nn.modules import C2PSA, C3k2, Conv, SPPF


class ConvBNAct(nn.Module):
    """Small explicit convolution block used where paper padding/dilation matters."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class DRFB(nn.Module):
    """Dilated Receptive Field Block.

    ``Conv_cat`` is not specified in the PDF. This implementation uses a 1x1
    compression from C concatenated channels to C/2, followed by the explicitly
    stated C/2-to-C projection and identity addition.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 2,
        dilations: Sequence[int] = (2, 3),
        branch_groups: Sequence[int] = (1, 1),
    ) -> None:
        super().__init__()
        if channels % reduction:
            raise ValueError("channels must be divisible by reduction")
        if len(dilations) != 2:
            raise ValueError("the paper specifies exactly two DRFB dilation branches")
        if len(branch_groups) != len(dilations):
            raise ValueError("branch_groups must match dilations")
        hidden = channels // reduction
        self.reduce = ConvBNAct(channels, hidden, 1)
        self.branches = nn.ModuleList(
            ConvBNAct(hidden, hidden, 3, padding=int(d), dilation=int(d), groups=int(g))
            for d, g in zip(dilations, branch_groups)
        )
        self.compress = ConvBNAct(hidden * len(dilations), hidden, 1)
        self.restore = ConvBNAct(hidden, channels, 1, activation=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        reduced = self.reduce(x)
        context = self.compress(torch.cat([branch(reduced) for branch in self.branches], dim=1))
        return self.act(x + self.restore(context))


class YOLO11DRFBBackbone(nn.Module):
    """YOLO11n P3-P5 backbone with early DRFB insertions.

    YOLO11n's published nano scaling is materialized directly (channels
    16/32/64/128/128/256 and one repeat after 0.5 depth scaling). The PDF does
    not disclose exact DRFB insertion indices; this implementation places them
    on P3 and P4, preserving both reported spatial levels without paying a
    disproportionate stride-4 cost.
    """

    out_channels: Tuple[int, int, int] = (128, 128, 256)

    def __init__(self) -> None:
        super().__init__()
        self.p1 = Conv(3, 16, 3, 2)
        self.p2_down = Conv(16, 32, 3, 2)
        self.p2 = C3k2(32, 64, n=1, c3k=False, e=0.25)

        self.p3_down = Conv(64, 64, 3, 2)
        self.p3 = C3k2(64, 128, n=1, c3k=False, e=0.25)
        # Group counts are an implementation choice used to match the paper's
        # 8.82-GFLOP deployment budget; the PDF does not specify convolution groups.
        self.drfb_p3 = DRFB(128, branch_groups=(1, 2))

        self.p4_down = Conv(128, 128, 3, 2)
        self.p4 = C3k2(128, 128, n=1, c3k=True, e=0.5)
        self.drfb_p4 = DRFB(128, branch_groups=(2, 2))

        self.p5_down = Conv(128, 256, 3, 2)
        self.p5 = C3k2(256, 256, n=1, c3k=True, e=0.5)
        self.sppf = SPPF(256, 256, k=5)
        self.psa = C2PSA(256, 256, n=1, e=0.5)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.p1(x)
        x = self.p2_down(x)
        x = self.p2(x)
        p3 = self.drfb_p3(self.p3(self.p3_down(x)))
        p4 = self.drfb_p4(self.p4(self.p4_down(p3)))
        p5 = self.psa(self.sppf(self.p5(self.p5_down(p4))))
        return p3, p4, p5


class FPNPAN(nn.Module):
    """YOLO11-style top-down FPN and bottom-up PAN for aligned router features."""

    out_channels: Tuple[int, int, int] = (64, 128, 256)

    def __init__(self, aligned_channels: int = 64) -> None:
        super().__init__()
        c = int(aligned_channels)
        self.top_p4 = C3k2(c * 2, 128, n=1, c3k=False, e=0.5)
        self.top_p3 = C3k2(c + 128, 64, n=1, c3k=False, e=0.5)
        self.down_p3 = Conv(64, 64, 3, 2)
        self.pan_p4 = C3k2(64 + 128, 128, n=1, c3k=False, e=0.5)
        self.down_p4 = Conv(128, 128, 3, 2)
        self.pan_p5 = C3k2(128 + c, 256, n=1, c3k=True, e=0.5)

    def forward(self, pyramid: Sequence[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        if len(pyramid) != 3:
            raise ValueError("FPNPAN expects P3, P4 and P5")
        p3, p4, p5 = pyramid
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        td4 = self.top_p4(torch.cat((p5_up, p4), dim=1))
        td4_up = F.interpolate(td4, size=p3.shape[-2:], mode="nearest")
        out3 = self.top_p3(torch.cat((td4_up, p3), dim=1))
        out4 = self.pan_p4(torch.cat((self.down_p3(out3), td4), dim=1))
        out5 = self.pan_p5(torch.cat((self.down_p4(out4), p5), dim=1))
        return out3, out4, out5


class SeparableTransform(nn.Module):
    """Lightweight expert transformation used for an otherwise unspecified FACH J_j."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = ConvBNAct(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class FeatureAwareCoupling(nn.Module):
    """One FACH level with channel conditioning, dynamic experts and identity path."""

    def __init__(self, channels: int, num_experts: int = 4, coupled_kernel: int = 1) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        self.channels = channels
        self.num_experts = num_experts
        self.channel_gate = nn.Linear(channels, channels)
        self.expert_gate = nn.Linear(channels, num_experts)
        self.experts = nn.ModuleList(SeparableTransform(channels) for _ in range(num_experts))
        if coupled_kernel not in (1, 3):
            raise ValueError("coupled_kernel must be 1 or 3")
        self.coupled = Conv(channels, channels, coupled_kernel, 1)

    def forward(self, x: Tensor) -> Tensor:
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        channel_weight = torch.sigmoid(self.channel_gate(pooled)).view(x.shape[0], self.channels, 1, 1)
        conditioned = x * channel_weight
        routing = torch.softmax(self.expert_gate(pooled), dim=-1)
        mixture = torch.zeros_like(x)
        for index, expert in enumerate(self.experts):
            mixture = mixture + routing[:, index].view(-1, 1, 1, 1) * expert(conditioned)
        return self.coupled(x + mixture)


class FACH(nn.Module):
    """Apply the Feature-Aware Coupled Head transform to all three neck levels."""

    def __init__(
        self,
        channels: Iterable[int] = (64, 128, 256),
        num_experts: int = 3,
        coupled_kernels: Sequence[int] = (1, 1, 3),
    ) -> None:
        super().__init__()
        channel_list = [int(c) for c in channels]
        if len(channel_list) != len(coupled_kernels):
            raise ValueError("coupled_kernels must match feature levels")
        self.levels = nn.ModuleList(
            FeatureAwareCoupling(c, num_experts, int(k)) for c, k in zip(channel_list, coupled_kernels)
        )

    def forward(self, features: Sequence[Tensor]) -> List[Tensor]:
        if len(features) != len(self.levels):
            raise ValueError("feature count does not match FACH levels")
        return [level(feature) for level, feature in zip(self.levels, features)]

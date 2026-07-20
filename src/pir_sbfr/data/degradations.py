"""Paper-faithful paired training and controlled evaluation degradations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


REFERENCE_METADATA = (1.0, 0.5, 30.0)


def sigma_from_mtf(mtf: float) -> float:
    """Gaussian PSF sigma ``sqrt(-2 log(q)) / pi`` from the paper."""
    if not 0.0 < float(mtf) <= 1.0:
        raise ValueError("MTF must lie in (0, 1]")
    return math.sqrt(-2.0 * math.log(float(mtf))) / math.pi


def gaussian_kernel2d(sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Normalized isotropic Gaussian kernel with width ``2*ceil(3*sigma)+1``."""
    if float(sigma) <= 0.0:
        return torch.ones((1, 1), device=device, dtype=dtype)
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    one_dimensional = torch.exp(-(coordinates**2) / (2.0 * float(sigma) ** 2))
    one_dimensional = one_dimensional / one_dimensional.sum()
    return one_dimensional[:, None] * one_dimensional[None, :]


def torch_gaussian_blur(image: Tensor, mtf: float) -> Tensor:
    """Blur one ``[C,H,W]`` image with the paper's isotropic Gaussian PSF."""
    sigma = sigma_from_mtf(mtf)
    kernel = gaussian_kernel2d(sigma, image.device, image.dtype)
    channels = image.shape[0]
    weight = kernel.view(1, 1, *kernel.shape).expand(channels, 1, -1, -1)
    padding = kernel.shape[-1] // 2
    # Reflection avoids introducing a dark border not described by the paper.
    padded = F.pad(image.unsqueeze(0), (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(padded, weight, groups=channels).squeeze(0)


def torch_poisson_noise(image: Tensor, snr_db: float, generator: Optional[torch.Generator] = None) -> Tensor:
    """Apply paper shot noise ``Poisson(lambda*x)/lambda`` to one image."""
    mean = image.mean()
    mean_square = image.square().mean()
    if float(mean_square.detach()) <= torch.finfo(image.dtype).eps:
        return image.clone()
    rate = mean * (10.0 ** (float(snr_db) / 10.0)) / mean_square.clamp_min(torch.finfo(image.dtype).eps)
    rate = rate.clamp_min(torch.finfo(image.dtype).eps)
    noisy = torch.poisson((rate * image).clamp_min(0.0), generator=generator) / rate
    return noisy.clamp_(0.0, 1.0)


@dataclass
class PairedBatch:
    """Clean/degraded views and their acquisition descriptors."""

    clean: Tensor
    degraded: Tensor
    metadata_clean: Tensor
    metadata_degraded: Tensor
    availability: Tensor
    modes: Tuple[str, ...]


class PairedDegradationGenerator:
    """Create deterministic 1:1 clean/degraded training pairs.

    A sample's random state is a stable hash of paper seed, epoch and image
    identifier. The same metadata-dropout mask is used for both paired views;
    this isolates degradation consistency from an extra missingness difference.
    The PDF does not state whether paired masks are shared, so this convention is
    explicitly documented.
    """

    def __init__(
        self,
        seed: int = 2023,
        metadata_dropout: float = 0.25,
        blur_range: Sequence[float] = (0.15, 0.45),
        snr_range: Sequence[float] = (10.0, 28.0),
        mode_probabilities: Sequence[float] = (0.35, 0.35, 0.30),
    ) -> None:
        if not 0.0 <= metadata_dropout <= 1.0:
            raise ValueError("metadata_dropout must be in [0,1]")
        if len(mode_probabilities) != 3 or not math.isclose(sum(mode_probabilities), 1.0, abs_tol=1e-7):
            raise ValueError("mode probabilities [blur, noise, joint] must sum to one")
        self.seed = int(seed)
        self.metadata_dropout = float(metadata_dropout)
        self.blur_range = (float(blur_range[0]), float(blur_range[1]))
        self.snr_range = (float(snr_range[0]), float(snr_range[1]))
        self.mode_probabilities = tuple(float(x) for x in mode_probabilities)

    def _sample_seed(self, epoch: int, key: str) -> int:
        payload = f"{self.seed}|{int(epoch)}|{key}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF

    @staticmethod
    def _uniform(generator: torch.Generator, low: float, high: float, device: torch.device) -> float:
        value = torch.rand((), generator=generator, device=device)
        return low + (high - low) * float(value)

    def __call__(self, images: Tensor, sample_keys: Sequence[str], epoch: int) -> PairedBatch:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch,3,height,width]")
        if len(sample_keys) != images.shape[0]:
            raise ValueError("sample_keys must contain one identifier per image")
        if images.min() < 0 or images.max() > 1:
            raise ValueError("paired degradations expect images normalized to [0,1]")

        reference = images.new_tensor(REFERENCE_METADATA).view(1, 3)
        metadata_clean = reference.expand(images.shape[0], -1).clone()
        metadata_degraded = metadata_clean.clone()
        availability = torch.ones_like(metadata_clean)
        degraded = images.clone()
        modes = []

        blur_cutoff = self.mode_probabilities[0]
        noise_cutoff = blur_cutoff + self.mode_probabilities[1]
        for index, key in enumerate(sample_keys):
            generator = torch.Generator(device=images.device)
            generator.manual_seed(self._sample_seed(epoch, str(key)))
            choice = float(torch.rand((), generator=generator, device=images.device))
            if choice < blur_cutoff:
                mode = "blur"
            elif choice < noise_cutoff:
                mode = "noise"
            else:
                mode = "joint"

            if mode in {"blur", "joint"}:
                mtf = self._uniform(generator, *self.blur_range, device=images.device)
                metadata_degraded[index, 1] = mtf
                degraded[index] = torch_gaussian_blur(degraded[index], mtf)
            if mode in {"noise", "joint"}:
                snr = self._uniform(generator, *self.snr_range, device=images.device)
                metadata_degraded[index, 2] = snr
                degraded[index] = torch_poisson_noise(degraded[index], snr, generator)

            availability[index] = (
                torch.rand(3, generator=generator, device=images.device) >= self.metadata_dropout
            ).to(dtype=images.dtype)
            modes.append(mode)

        return PairedBatch(
            clean=images,
            degraded=degraded,
            metadata_clean=metadata_clean,
            metadata_degraded=metadata_degraded,
            availability=availability,
            modes=tuple(modes),
        )


@dataclass(frozen=True)
class DegradationCondition:
    """Controlled GSD-MTF-SNR grid condition."""

    gsd: float = 1.0
    mtf: float = 0.5
    snr: float = 30.0

    @property
    def metadata(self) -> Tuple[float, float, float]:
        return float(self.gsd), float(self.mtf), float(self.snr)

    @property
    def name(self) -> str:
        return f"gsd{self.gsd:g}_mtf{self.mtf:g}_snr{self.snr:g}"


def _resize_gsd(image: np.ndarray, ratio: float) -> np.ndarray:
    if ratio <= 1.0:
        return image
    height, width = image.shape[:2]
    down_h = max(1, int(round(height / ratio)))
    down_w = max(1, int(round(width / ratio)))
    # The PDF explicitly specifies antialiased bicubic reduction followed by
    # bicubic reconstruction. PyTorch exposes the antialias switch directly,
    # unlike OpenCV's cubic resize API.
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
    reduced = F.interpolate(
        tensor,
        size=(down_h, down_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    restored = F.interpolate(
        reduced,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    )
    return restored.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)


def _numpy_gaussian_blur(image: np.ndarray, mtf: float) -> np.ndarray:
    sigma = sigma_from_mtf(mtf)
    width = 2 * int(math.ceil(3.0 * sigma)) + 1
    return cv2.GaussianBlur(image, (width, width), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)


def _numpy_poisson_noise(image: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    mean = float(image.mean())
    mean_square = float(np.square(image).mean())
    if mean_square <= np.finfo(np.float32).eps:
        return image.copy()
    rate = mean * (10.0 ** (float(snr_db) / 10.0)) / max(mean_square, np.finfo(np.float32).eps)
    return np.clip(rng.poisson(rate * np.clip(image, 0.0, 1.0)) / rate, 0.0, 1.0).astype(np.float32)


def controlled_degradation(
    image: np.ndarray,
    condition: DegradationCondition,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply the complete controlled transformation in fixed paper order.

    The reference point (1, 0.50, 30 dB) is treated as the unmodified clean
    image, which is necessary for the clean grid cell to equal the paper's main
    clean-set result. Operators are applied only below/above that reference.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an HWC RGB array")
    rng = rng or np.random.default_rng(0)
    original_dtype = image.dtype
    result = image.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        result /= float(np.iinfo(original_dtype).max)
    result = np.clip(result, 0.0, 1.0)
    result = _resize_gsd(result, float(condition.gsd))
    if float(condition.mtf) < REFERENCE_METADATA[1]:
        result = _numpy_gaussian_blur(result, float(condition.mtf))
    if float(condition.snr) < REFERENCE_METADATA[2]:
        result = _numpy_poisson_noise(result, float(condition.snr), rng)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def disk_psf(image: np.ndarray, radius: int = 3) -> np.ndarray:
    size = radius * 2 + 1
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def motion_psf(image: np.ndarray, length: int = 9, angle_degrees: float = 0.0) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_degrees, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length), flags=cv2.INTER_LINEAR)
    kernel /= max(float(kernel.sum()), np.finfo(np.float32).eps)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def anisotropic_psf(
    image: np.ndarray,
    sigma_x: float = 2.5,
    sigma_y: float = 0.6,
    angle_degrees: float = 0.0,
) -> np.ndarray:
    radius = int(math.ceil(3.0 * max(sigma_x, sigma_y)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coordinates, coordinates)
    theta = math.radians(angle_degrees)
    xr = xx * math.cos(theta) + yy * math.sin(theta)
    yr = -xx * math.sin(theta) + yy * math.cos(theta)
    kernel = np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2)).astype(np.float32)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def speckle_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(image + image * rng.normal(0.0, sigma, image.shape), 0.0, 1.0).astype(np.float32)


def stripe_read_noise(
    image: np.ndarray,
    amplitude: float,
    read_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # Stripe orientation/frequency/phase are absent from the PDF. This explicit
    # The official held-out evaluator uses a vertical sinusoidal realization.
    width = image.shape[1]
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    stripe = amplitude * np.sin(np.linspace(0.0, 8.0 * math.pi, width, dtype=np.float32) + phase)
    read = rng.normal(0.0, read_sigma, image.shape).astype(np.float32)
    return np.clip(image + stripe[None, :, None] + read, 0.0, 1.0).astype(np.float32)

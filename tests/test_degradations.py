import numpy as np
import torch

from pir_sbfr.data.degradations import (
    DegradationCondition,
    PairedDegradationGenerator,
    controlled_degradation,
    sigma_from_mtf,
    torch_gaussian_blur,
)


def test_paired_degradation_is_deterministic_and_keeps_reference_gsd():
    images = torch.rand(4, 3, 32, 32)
    generator = PairedDegradationGenerator(seed=2023)
    first = generator(images, ["a", "b", "c", "d"], epoch=7)
    second = generator(images, ["a", "b", "c", "d"], epoch=7)
    assert first.modes == second.modes
    assert torch.equal(first.degraded, second.degraded)
    assert torch.equal(first.availability, second.availability)
    torch.testing.assert_close(first.metadata_clean[:, 0], torch.ones(4))
    torch.testing.assert_close(first.metadata_degraded[:, 0], torch.ones(4))
    assert 0.0 <= float(first.degraded.min()) <= float(first.degraded.max()) <= 1.0


def test_controlled_reference_is_identity():
    image = np.random.default_rng(2).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    result = controlled_degradation(image, DegradationCondition())
    np.testing.assert_allclose(result, image.astype(np.float32) / 255.0, atol=0, rtol=0)


def test_degraded_condition_changes_image():
    image = np.random.default_rng(3).random((32, 32, 3), dtype=np.float32)
    result = controlled_degradation(
        image,
        DegradationCondition(gsd=2.0, mtf=0.15, snr=10.0),
        np.random.default_rng(4),
    )
    assert result.shape == image.shape
    assert not np.allclose(result, image)
    assert 0 <= result.min() <= result.max() <= 1
    assert sigma_from_mtf(0.15) > sigma_from_mtf(0.5)


def test_unit_mtf_blur_is_identity():
    image = torch.rand(3, 16, 16)
    torch.testing.assert_close(torch_gaussian_blur(image, 1.0), image)

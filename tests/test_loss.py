import torch

from pir_sbfr.models.loss import object_scale_distribution, scale_kl_divergence


def test_dior_scale_distribution_counts_three_groups():
    batch = {
        "bboxes": torch.tensor(
            [
                [0.5, 0.5, 10 / 640, 10 / 640],
                [0.5, 0.5, 40 / 640, 40 / 640],
                [0.5, 0.5, 100 / 640, 100 / 640],
            ]
        ),
        "batch_idx": torch.tensor([0.0, 0.0, 0.0]),
    }
    target = object_scale_distribution(batch, 1, (640, 640), mode="dior")
    torch.testing.assert_close(target, torch.full((1, 3), 1 / 3), atol=1e-6, rtol=1e-6)
    assert float(scale_kl_divergence(target, target)) == 0.0


def test_empty_image_target_is_uniform():
    batch = {"bboxes": torch.empty(0, 4), "batch_idx": torch.empty(0)}
    target = object_scale_distribution(batch, 2, (640, 640), mode="dior")
    torch.testing.assert_close(target, torch.full((2, 3), 1 / 3))

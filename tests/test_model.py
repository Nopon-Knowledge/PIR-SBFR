import torch

from pir_sbfr.models.detector import PIRSBFRModel


def _batch():
    return {
        "img": torch.rand(2, 3, 64, 64),
        "img_degraded": torch.rand(2, 3, 64, 64),
        "metadata": torch.tensor([[1.0, 0.5, 30.0], [1.0, 0.5, 30.0]]),
        "metadata_degraded": torch.tensor([[1.0, 0.2, 30.0], [1.0, 0.5, 15.0]]),
        "availability": torch.ones(2, 3),
        "availability_degraded": torch.ones(2, 3),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[1.0], [2.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25], [0.4, 0.4, 0.15, 0.2]]),
    }


def test_training_forward_and_backward():
    model = PIRSBFRModel(nc=3).train()
    loss, items = model(_batch())
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert items.shape == (5,) and torch.isfinite(items).all()
    loss.backward()
    assert model.neck.visual.gate.weight.grad is not None


def test_inference_shape_and_paper_parameter_budget():
    model = PIRSBFRModel(nc=20).eval()
    with torch.no_grad():
        decoded, raw = model(torch.rand(1, 3, 64, 64))
    assert decoded.shape == (1, 24, 84)
    assert [x.shape[-2:] for x in raw] == [torch.Size([8, 8]), torch.Size([4, 4]), torch.Size([2, 2])]
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert abs(parameters - 3_942_000) < 5_000


def test_no_auxiliary_ablation_trains_without_dead_bypass():
    model = PIRSBFRModel(
        nc=3,
        router_config={"p5_bypass": False},
        lambda_scale=0.0,
        lambda_consistency=0.0,
    ).train()
    loss, items = model(_batch())
    assert model.neck.p5_projection is None
    torch.testing.assert_close(items[-2:], torch.zeros(2))
    loss.backward()

import torch

from pir_sbfr.models.router import PIRSBFRNeck


def _pyramid(batch=2):
    return (
        torch.randn(batch, 128, 8, 8),
        torch.randn(batch, 128, 4, 4),
        torch.randn(batch, 256, 2, 2),
    )


def test_router_outputs_and_normalization():
    neck = PIRSBFRNeck().eval()
    with torch.no_grad():
        outputs, aux = neck(_pyramid(), torch.tensor([[1.0, 0.5, 30.0]] * 2), torch.ones(2, 3))
    assert [tuple(x.shape) for x in outputs] == [(2, 64, 8, 8), (2, 128, 4, 4), (2, 256, 2, 2)]
    torch.testing.assert_close(aux["weights"].sum(-1), torch.ones(2), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(aux["rho_phy"], torch.ones(2, 3))


def test_physical_only_degradation_moves_mass_from_p3():
    neck = PIRSBFRNeck(use_visual=False).eval()
    features = _pyramid(batch=1)
    with torch.no_grad():
        _, clear = neck(features, torch.tensor([[1.0, 0.5, 30.0]]), torch.ones(1, 3))
        _, degraded = neck(features, torch.tensor([[3.0, 0.15, 10.0]]), torch.ones(1, 3))
    torch.testing.assert_close(clear["weights"], torch.full((1, 3), 1 / 3), atol=1e-6, rtol=1e-6)
    assert degraded["weights"][0, 0] < clear["weights"][0, 0]
    assert degraded["weights"][0, 2] > clear["weights"][0, 2]


def test_missing_mask_ignores_invalid_values():
    neck = PIRSBFRNeck(use_visual=False).eval()
    with torch.no_grad():
        _, aux = neck(
            _pyramid(batch=1),
            torch.tensor([[0.0, float("nan"), -1000.0]]),
            torch.zeros(1, 3),
        )
    torch.testing.assert_close(aux["rho_phy"], torch.ones(1, 3))


def test_disabling_p5_bypass_removes_projection_parameters():
    with_bypass = PIRSBFRNeck(p5_bypass=True)
    without_bypass = PIRSBFRNeck(p5_bypass=False)
    assert without_bypass.p5_projection is None
    difference = sum(p.numel() for p in with_bypass.parameters()) - sum(p.numel() for p in without_bypass.parameters())
    assert difference == 66_048

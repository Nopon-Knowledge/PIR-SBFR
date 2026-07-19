import math

import torch

from pir_sbfr.models.physical import PhysicalReliabilityPrior


def test_reference_and_missing_are_neutral():
    prior = PhysicalReliabilityPrior()
    metadata = torch.tensor([[1.0, 0.5, 30.0], [0.0, float("nan"), -999.0]])
    availability = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    actual = prior(metadata, availability)
    torch.testing.assert_close(actual, torch.ones_like(actual))


def test_gsd_two_matches_closed_form_and_is_monotonic():
    prior = PhysicalReliabilityPrior()
    actual = prior(torch.tensor([[2.0, 0.5, 30.0]]), torch.ones(1, 3))[0]
    expected = torch.tensor([0.5, math.sqrt(0.5), 0.5**0.25])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert actual[0] < actual[1] < actual[2]


def test_better_than_reference_is_clamped_to_one():
    prior = PhysicalReliabilityPrior()
    actual = prior(torch.tensor([[0.5, 0.8, 40.0]]), torch.ones(1, 3))
    torch.testing.assert_close(actual, torch.ones_like(actual))

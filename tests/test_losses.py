import numpy as np
import pytest
import torch

from src.tsfm_pretraining import losses as L


def test_gini_known_values():
    assert L.gini_coefficient(np.array([5.0, 5.0, 5.0])) == 0.0
    assert L.gini_coefficient(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(0.25)
    assert L.gini_coefficient(np.array([0.0, 0.0, 0.0, 10.0])) == pytest.approx(0.75)
    assert L.gini_coefficient(np.array([7.0])) == 0.0


def test_gini_rejects_negative_values():
    with pytest.raises(ValueError):
        L.gini_coefficient(np.array([1.0, -1.0]))


def test_masked_mse_requires_matching_ndim():
    pred = torch.randn(4, 1, 8)
    target = torch.randn(4, 1, 8)
    bad_mask = torch.ones(4, 8)  # missing the channel dim
    with pytest.raises(ValueError, match="ndim"):
        L.masked_mse(pred, target, bad_mask, reduction="none")


def test_masked_mse_matches_manual_computation():
    pred = torch.randn(4, 1, 8)
    target = torch.randn(4, 1, 8)
    mask = torch.ones(4, 1, 8)
    mask[:, :, 4:] = 0
    per_example = L.masked_mse(pred, target, mask, reduction="none")
    manual = ((pred[:, :, :4] - target[:, :, :4]) ** 2).mean(dim=(1, 2))
    assert torch.allclose(per_example, manual)


def test_pinball_zero_at_perfect_prediction():
    pred = torch.zeros(5, 3)
    target = torch.zeros(5)
    loss = L.pinball_loss(pred, target, [0.1, 0.5, 0.9], reduction="none")
    assert torch.allclose(loss, torch.zeros(5))


def test_dispersion_metrics_and_pooled_mean_diverge_for_imbalanced_sources():
    """Mirrors the plan's requirement that the pooled global metric and the
    unweighted per-source mean be genuinely different for an imbalanced
    (natural-mixture) validation set."""
    per_example_error = np.concatenate([np.full(90, 1.0), np.full(10, 10.0)])
    source_ids = np.array(["big"] * 90 + ["small"] * 10)
    per_source = L.group_mean_by_source(per_example_error, source_ids)
    metrics = L.dispersion_metrics(per_source)
    pooled = L.pooled_mean(per_example_error)

    assert metrics["n_sources"] == 2
    assert metrics["unweighted_mean"] == pytest.approx((1.0 + 10.0) / 2)
    assert pooled == pytest.approx((90 * 1.0 + 10 * 10.0) / 100)
    assert metrics["unweighted_mean"] != pytest.approx(pooled)


def test_log_mse_auc_and_steps_to_threshold():
    steps = np.array([0, 100, 500, 1000, 3000])
    mse = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
    auc_2000 = L.log_mse_auc(steps, mse, cutoff_step=2000)
    auc_500 = L.log_mse_auc(steps, mse, cutoff_step=500)
    assert auc_2000 > auc_500  # more area accumulated over a longer window
    assert L.steps_to_threshold(steps, mse, 2.0) == 500
    assert L.steps_to_threshold(steps, mse, 0.01) is None

import numpy as np
import pytest
import torch

from src.data import seasonality
from src.losses import pointwise, quantile
from src.metrics import convergence, forecast, inequality
from src.training import gradients


def test_gini_known_values():
    assert inequality.gini_coefficient(np.array([5.0, 5.0, 5.0])) == 0.0
    assert inequality.gini_coefficient(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(
        0.25
    )
    assert inequality.gini_coefficient(
        np.array([0.0, 0.0, 0.0, 10.0])
    ) == pytest.approx(0.75)
    assert inequality.gini_coefficient(np.array([7.0])) == 0.0


def test_gini_rejects_negative_values():
    with pytest.raises(ValueError):
        inequality.gini_coefficient(np.array([1.0, -1.0]))


def test_masked_mse_requires_matching_ndim():
    pred = torch.randn(4, 1, 8)
    target = torch.randn(4, 1, 8)
    bad_mask = torch.ones(4, 8)  # missing the channel dim
    with pytest.raises(ValueError, match="ndim"):
        pointwise.masked_mse(pred, target, bad_mask, reduction="none")


def test_masked_mse_matches_manual_computation():
    pred = torch.randn(4, 1, 8)
    target = torch.randn(4, 1, 8)
    mask = torch.ones(4, 1, 8)
    mask[:, :, 4:] = 0
    per_example = pointwise.masked_mse(pred, target, mask, reduction="none")
    manual = ((pred[:, :, :4] - target[:, :, :4]) ** 2).mean(dim=(1, 2))
    assert torch.allclose(per_example, manual)


def test_pinball_zero_at_perfect_prediction():
    pred = torch.zeros(5, 3)
    target = torch.zeros(5)
    loss = quantile.pinball_loss(pred, target, [0.1, 0.5, 0.9], reduction="none")
    assert torch.allclose(loss, torch.zeros(5))


def test_dispersion_metrics_and_pooled_mean_diverge_for_imbalanced_sources():
    """Mirrors the plan's requirement that the pooled global metric and the
    unweighted per-source mean be genuinely different for an imbalanced
    (natural-mixture) validation set."""
    per_example_error = np.concatenate([np.full(90, 1.0), np.full(10, 10.0)])
    source_ids = np.array(["big"] * 90 + ["small"] * 10)
    per_source = inequality.group_mean_by_source(per_example_error, source_ids)
    metrics = inequality.dispersion_metrics(per_source)
    pooled = inequality.pooled_mean(per_example_error)

    assert metrics["n_sources"] == 2
    assert metrics["unweighted_mean"] == pytest.approx((1.0 + 10.0) / 2)
    assert pooled == pytest.approx((90 * 1.0 + 10 * 10.0) / 100)
    assert metrics["unweighted_mean"] != pytest.approx(pooled)


def test_log_mse_auc_and_steps_to_threshold():
    steps = np.array([0, 100, 500, 1000, 3000])
    mse = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
    auc_2000 = convergence.log_mse_auc(steps, mse, cutoff_step=2000)
    auc_500 = convergence.log_mse_auc(steps, mse, cutoff_step=500)
    assert auc_2000 > auc_500  # more area accumulated over a longer window
    assert convergence.steps_to_threshold(steps, mse, 2.0) == 500
    assert convergence.steps_to_threshold(steps, mse, 0.01) is None


def test_mase_is_invariant_to_scale():
    """The controlled scale intervention multiplies a window by b, which scales
    the MASE numerator and its seasonal-naive denominator equally. This is the
    property that makes MASE a fair metric across scale assignments."""
    torch.manual_seed(0)
    context = torch.randn(4, 64).cumsum(dim=1)
    valid = torch.ones(4, 64)
    periods = torch.tensor([24, 7, 24, 7])
    pred = context + 0.3 * torch.randn(4, 64)

    def mase(b: float) -> torch.Tensor:
        numerator = pointwise.masked_mae(pred * b, context * b, valid, reduction="none")
        return numerator / forecast.seasonal_naive_mae(context * b, valid, periods)

    assert torch.allclose(mase(1.0), mase(10.0), atol=1e-6)


def test_seasonal_naive_mae_flags_constant_windows():
    context = torch.stack([torch.ones(32), torch.arange(32).float()])
    naive = forecast.seasonal_naive_mae(
        context, torch.ones(2, 32), torch.tensor([7, 7])
    )
    assert torch.isnan(naive[0])  # constant -> zero denominator
    assert naive[1] == pytest.approx(7.0)


def test_seasonal_naive_mae_falls_back_to_lag_one_when_period_too_long():
    """ "4S" implies a 21,600-step daily cycle against a 512-step context;
    falling back to lag 1 keeps the dataset in the dispersion metrics."""
    context = torch.arange(32).float().unsqueeze(0)
    valid = torch.ones(1, 32)
    too_long = forecast.seasonal_naive_mae(context, valid, torch.tensor([100]))
    lag_one = forecast.seasonal_naive_mae(context, valid, torch.tensor([1]))
    assert too_long[0] == pytest.approx(lag_one[0])
    assert too_long[0] == pytest.approx(1.0)


def test_seasonal_period_handles_multipliers_and_anchors():
    assert seasonality.seasonal_period("H") == 24
    assert seasonality.seasonal_period("5T") == 288
    assert seasonality.seasonal_period("D") == 7
    assert seasonality.seasonal_period("M") == 12
    assert seasonality.seasonal_period("Q-DEC") == 4
    assert seasonality.seasonal_period("W-SUN") == 1
    assert seasonality.seasonal_period("A-DEC") == 1


def test_aggregation_drops_unusable_mase_windows():
    errors = np.array([np.nan, 2.0, 4.0, np.nan])
    sources = np.array(["a", "a", "b", "b"])
    per_source = inequality.group_mean_by_source(errors, sources)
    assert per_source == {"a": 2.0, "b": 4.0}
    assert inequality.pooled_mean(errors) == pytest.approx(3.0)
    assert (
        inequality.dispersion_metrics({"a": float("nan"), "b": 3.0})["n_sources"] == 1
    )


def test_normalized_space_mase_equals_original_space_mase():
    """MASE computed in normalized space is identical to MASE in original
    space: the affine normalization (x - a) / b divides numerator and
    denominator by the same b, and the shift a cancels in both because each is
    built from differences. So "nMASE" is not a distinct metric from MASE."""
    torch.manual_seed(0)
    context = torch.randn(4, 64).cumsum(dim=1)
    valid = torch.ones(4, 64)
    periods = torch.tensor([24, 24, 7, 7])
    pred = context + 0.3 * torch.randn(4, 64)

    a = context.mean(dim=1, keepdim=True)
    b = context.std(dim=1, keepdim=True)
    normalized_context = (context - a) / b
    normalized_pred = (pred - a) / b

    mase = pointwise.masked_mae(
        pred, context, valid, reduction="none"
    ) / forecast.seasonal_naive_mae(context, valid, periods)
    nmase = pointwise.masked_mae(
        normalized_pred, normalized_context, valid, reduction="none"
    ) / forecast.seasonal_naive_mae(normalized_context, valid, periods)

    assert torch.allclose(mase, nmase, atol=1e-6)


def test_safe_gradient_clipping_handles_norm_square_overflow():
    parameter = torch.tensor(1.0, requires_grad=True)
    loss = (parameter * 1e15).square()

    metrics = gradients.backward_with_safe_gradient_clipping(loss, [parameter], 1.0)

    assert np.isclose(metrics["total_norm_before_clip"], 2e30, rtol=1e-6)
    assert np.isclose(metrics["total_norm_after_clip"], 1.0, rtol=1e-6)
    assert np.isclose(float(parameter.grad), 1.0, rtol=1e-6)
    assert metrics["clipped"]


def test_safe_gradient_clipping_restores_unclipped_gradient():
    parameter = torch.tensor(2.0, requires_grad=True)
    loss = parameter.square()

    metrics = gradients.backward_with_safe_gradient_clipping(loss, [parameter], 10.0)

    assert np.isclose(metrics["total_norm_before_clip"], 4.0)
    assert np.isclose(metrics["total_norm_after_clip"], 4.0)
    assert np.isclose(float(parameter.grad), 4.0)
    assert not metrics["clipped"]


def test_pinball_loss_matches_uni2ts_packed_quantile_mae():
    """Moirai 2.0's objective: pinball averaged over levels, no factor of 2."""
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    torch.manual_seed(0)
    pred = torch.randn(6, len(levels), 11, dtype=torch.float64)
    target = torch.randn(6, 11, dtype=torch.float64)
    valid = torch.ones(6, 11, dtype=torch.float64)

    # Transcribed from uni2ts/loss/packed/quantile.py PackedQuantileMAELoss.
    q = torch.tensor(levels, dtype=torch.float64).view(1, -1, 1)
    expanded = target.unsqueeze(1).expand_as(pred)
    errors = (pred - expanded).abs()
    reference = torch.where(expanded > pred, q * errors, (1 - q) * errors)
    reference = reference.mean(dim=-2).mean(dim=-1)

    actual = quantile.pinball_loss(
        pred.transpose(1, 2), target, levels, valid=valid, reduction="none"
    )
    assert torch.allclose(actual, reference)


def test_crps_quantile_loss_matches_chronos2():
    """Chronos-2's objective: twice pinball, summed over levels."""
    levels = [0.1, 0.5, 0.9]
    torch.manual_seed(1)
    pred = torch.randn(4, len(levels), 7, dtype=torch.float64)
    target = torch.randn(4, 7, dtype=torch.float64)
    valid = torch.ones(4, 7, dtype=torch.float64)
    q = torch.tensor(levels, dtype=torch.float64)

    # Transcribed from chronos/chronos2/model.py _compute_loss.
    expanded = target.unsqueeze(1)
    reference = 2 * torch.abs(
        (expanded - pred) * ((expanded <= pred).double() - q.view(1, -1, 1))
    )
    reference = reference.mean(dim=-1).sum(dim=-1)

    assert torch.allclose(
        quantile.crps_quantile_loss(pred, target, valid, q), reference
    )


def test_the_two_quantile_conventions_differ_by_two_q():
    """Guards against collapsing them back into one function."""
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    torch.manual_seed(2)
    pred = torch.randn(5, len(levels), 9, dtype=torch.float64)
    target = torch.randn(5, 9, dtype=torch.float64)
    valid = torch.ones(5, 9, dtype=torch.float64)

    crps = quantile.crps_quantile_loss(
        pred, target, valid, torch.tensor(levels, dtype=torch.float64)
    )
    pinball = quantile.pinball_loss(
        pred.transpose(1, 2), target, levels, valid=valid, reduction="none"
    )
    assert torch.allclose(crps / pinball, torch.full_like(crps, 2.0 * len(levels)))

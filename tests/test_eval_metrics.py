"""Eval-harness loss and stability metrics.

The quantile-loss checks pin the numpy eval form to the torch training form
in src/losses/quantile.py, since the harness reports one CRPS column across
models that each trained on a different objective.
"""

import numpy as np
import pytest
import torch

from src.losses import quantile as tq
from src.metrics import stability
from src.metrics.eval_losses import quantile_loss, weighted_quantile_loss

QUANTILES = [0.1, 0.5, 0.9]


def test_quantile_loss_matches_torch_pinball_up_to_2q():
    """The eval form is the doubled sum-over-quantiles CRPS convention and
    the training form averages over quantiles, so they must differ by
    exactly 2Q. Drift here silently rescales every reported CRPS."""
    rng = np.random.default_rng(0)
    preds = rng.normal(size=(4, 8, 3)).astype(np.float64)
    targets = rng.normal(size=(4, 8)).astype(np.float64)

    ours = quantile_loss(preds, targets, QUANTILES, aggregate="mean")
    theirs = tq.pinball_loss(
        torch.from_numpy(preds), torch.from_numpy(targets), QUANTILES
    ).item()

    assert ours == pytest.approx(2 * len(QUANTILES) * theirs)


def test_quantile_loss_mask_excludes_padded_positions():
    rng = np.random.default_rng(1)
    preds = rng.normal(size=(2, 6, 3))
    targets = rng.normal(size=(2, 6))
    mask = np.ones((2, 6))
    mask[:, 4:] = 0.0

    masked = quantile_loss(preds, targets, QUANTILES, mask, aggregate="sum")
    truncated = quantile_loss(preds[:, :4], targets[:, :4], QUANTILES, aggregate="sum")
    assert masked == pytest.approx(truncated)


def test_quantile_loss_aggregate_none_keeps_element_shape():
    """excess_volatility composes three losses elementwise, so the
    unaggregated form must keep the quantile axis rather than reduce it."""
    preds = np.zeros((2, 5, 3))
    targets = np.zeros((2, 5))
    out = quantile_loss(preds, targets, QUANTILES, aggregate=None)
    assert out.shape == (2, 5, 3)


def test_perfect_forecast_scores_zero_loss():
    targets = np.arange(12, dtype=np.float64).reshape(2, 6)
    preds = np.repeat(targets[..., None], 3, axis=-1)
    assert quantile_loss(preds, targets, QUANTILES, aggregate="sum") == 0.0
    assert weighted_quantile_loss(preds, targets, QUANTILES) == 0.0


def test_weighted_quantile_loss_is_scale_free():
    """WQL normalizes by total absolute target, so scaling a series must not
    move it. This is the property that makes it comparable across the six
    suites, whose units differ by orders of magnitude."""
    rng = np.random.default_rng(2)
    targets = rng.uniform(1.0, 5.0, size=(3, 7))
    preds = targets[..., None] + rng.normal(scale=0.2, size=(3, 7, 3))

    base = weighted_quantile_loss(preds, targets, QUANTILES)
    scaled = weighted_quantile_loss(preds * 1000.0, targets * 1000.0, QUANTILES)
    assert base == pytest.approx(scaled)


def test_reshape_windows_by_date_groups_a_known_target_date():
    """Window t predicts dates t*stride .. t*stride+H-1, so with stride=1 the
    h-th column of date d holds the forecast made h steps ahead of d."""
    preds = np.arange(2 * 3 * 4, dtype=np.float64).reshape(1, 2, 3, 4)
    out = stability.reshape_windows_by_date(preds, stride=1)

    assert out.shape == (1, 4, 3, 4)
    # date 2 is reachable from window 0 at h=2 and window 1 at h=1
    assert np.allclose(out[0, 2, 2, :], preds[0, 0, 2, :])
    assert np.allclose(out[0, 2, 1, :], preds[0, 1, 1, :])
    # date 0 is only reachable from window 0 at h=0
    assert np.isnan(out[0, 0, 1, :]).all()


def test_stability_metrics_reject_non_overlapping_windows():
    preds = np.zeros((1, 3, 4, 1))
    for stride in (4, 5):
        with pytest.raises(ValueError, match="stability|coverage"):
            stability.forecast_percentage_change(preds, stride=stride)
        with pytest.raises(ValueError, match="stability|coverage"):
            stability.excess_volatility(
                np.zeros((1, 3, 4, 1)),
                np.zeros((1, 3, 4, 1, 3)),
                QUANTILES,
                stride=stride,
            )


def test_sfpc_is_zero_when_the_forecast_never_revises():
    """A model that predicts the same value for a date regardless of when it
    forecasts has no churn, which is the metric's fixed point."""
    constant = np.full((2, 4, 3, 1), 7.0)
    assert stability.forecast_percentage_change(constant, stride=1) == pytest.approx(
        0.0
    )


def test_sfpc_grows_with_revision_size():
    rng = np.random.default_rng(3)
    base = rng.uniform(5.0, 10.0, size=(2, 4, 3, 1))
    small = base + rng.normal(scale=0.01, size=base.shape)
    large = base + rng.normal(scale=1.0, size=base.shape)

    assert stability.forecast_percentage_change(
        small, stride=1
    ) < stability.forecast_percentage_change(large, stride=1)


def test_excess_volatility_penalizes_churn_that_buys_no_accuracy():
    """A forecast that converges on the truth as its creation date advances
    must score better than one that thrashes with strictly larger error.

    Revisions happen along the window axis, not the horizon axis, so the
    error pattern here varies with t. Pairing `before` with the newer rather
    than the older forecast inverts the accuracy term and fails this."""
    rng = np.random.default_rng(4)
    targets = rng.uniform(10.0, 20.0, size=(2, 6, 4, 1))
    window = np.arange(targets.shape[1])[None, :, None, None]

    converging = targets + 2.0 * (0.85**window)
    thrashing = targets + 2.0 * ((-1.0) ** window)
    assert np.abs(converging - targets).mean() < np.abs(thrashing - targets).mean()

    def to_q(x):
        return np.repeat(x[..., None], len(QUANTILES), axis=-1)

    ev_converging = stability.excess_volatility(
        targets, to_q(converging), QUANTILES, stride=1
    )
    ev_thrashing = stability.excess_volatility(
        targets, to_q(thrashing), QUANTILES, stride=1
    )
    assert ev_thrashing > ev_converging


def test_excess_volatility_is_near_zero_for_a_forecast_that_never_revises():
    """With no revision there is no cost and no accuracy change, so the two
    terms cancel and EV collapses regardless of how wrong the forecast is."""
    rng = np.random.default_rng(6)
    targets = rng.uniform(10.0, 20.0, size=(2, 5, 4, 1))
    date_offset = np.arange(targets.shape[1] + targets.shape[2] - 1)

    # every window predicts the same value for a given target date
    stable = np.empty_like(targets)
    for t in range(targets.shape[1]):
        for h in range(targets.shape[2]):
            stable[:, t, h, 0] = 100.0 + date_offset[t + h]
    stable_q = np.repeat(stable[..., None], len(QUANTILES), axis=-1)

    assert stability.excess_volatility(
        targets, stable_q, QUANTILES, stride=1
    ) == pytest.approx(0.0, abs=1e-9)


def test_excess_volatility_scaling_is_unit_free():
    rng = np.random.default_rng(5)
    targets = rng.uniform(1.0, 3.0, size=(2, 4, 3, 1))
    preds = np.repeat(
        (targets + rng.normal(scale=0.3, size=targets.shape))[..., None], 3, axis=-1
    )

    base = stability.excess_volatility(targets, preds, QUANTILES, stride=1)
    scaled = stability.excess_volatility(
        targets * 100.0, preds * 100.0, QUANTILES, stride=1
    )
    assert base == pytest.approx(scaled, rel=1e-6)

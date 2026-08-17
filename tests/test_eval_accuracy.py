"""Accuracy metrics for the eval harness.

The MASE checks matter most: a seasonal-naive forecast must score ~1.0, and
that is the property that catches a horizon misalignment, which otherwise
produces plausible-looking numbers rather than an obvious failure.
"""

import numpy as np
import pytest

from src.metrics import accuracy

QUANTILES = [0.1, 0.5, 0.9]


def _quantile_stack(point: np.ndarray) -> np.ndarray:
    return np.repeat(point[..., None], len(QUANTILES), axis=-1)


def _seasonal_series(n_series: int, length: int, period: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    season = rng.normal(size=(n_series, period))
    idx = np.arange(length)[None, :]
    return season[:, idx[0] % period] + rng.normal(scale=0.01, size=(n_series, length))


def test_perfect_forecast_scores_zero_everywhere():
    targets = np.array([[1.0, 2.0, 3.0, 4.0]])
    history = np.arange(1.0, 21.0).reshape(1, 20)
    out = accuracy.per_series_metrics(
        _quantile_stack(targets),
        targets,
        np.ones_like(targets),
        history,
        np.ones_like(history),
        QUANTILES,
        ["D"],
    )
    for name in ("mae", "mse", "nmse", "mase", "mape", "smape", "crps", "wql"):
        assert out[name][0] == pytest.approx(0.0), name


def test_seasonal_naive_forecast_scores_mase_near_one():
    """MASE is the horizon MAE over the in-sample seasonal-naive MAE, so a
    seasonal-naive forecast is the metric's unit by construction. A horizon
    offset by even one step breaks this badly on seasonal data."""
    period, length, horizon = 7, 70, 14
    series = _seasonal_series(16, length + horizon, period, seed=1)
    history, targets = series[:, :length], series[:, length:]
    # seasonal naive: repeat the last full season forward
    forecast = np.concatenate([history[:, -period:], history[:, -period:]], axis=1)[
        :, :horizon
    ]

    out = accuracy.per_series_metrics(
        _quantile_stack(forecast),
        targets,
        np.ones_like(targets),
        history,
        np.ones_like(history),
        QUANTILES,
        ["D"] * 16,
    )
    assert np.nanmean(out["mase"]) == pytest.approx(1.0, rel=0.15)


def test_misaligned_horizon_blows_up_mase():
    """The failure this suite exists to catch: forecasting the right shape
    one step out of phase still looks reasonable in raw MAE but must not
    pass as MASE ~ 1."""
    period, length, horizon = 7, 70, 14
    series = _seasonal_series(16, length + horizon + 1, period, seed=2)
    history, targets = series[:, :length], series[:, length : length + horizon]
    aligned = np.concatenate([history[:, -period:]] * 2, axis=1)[:, :horizon]
    shifted = np.concatenate([history[:, -period + 1 :]] * 3, axis=1)[:, :horizon]

    def mase(forecast):
        out = accuracy.per_series_metrics(
            _quantile_stack(forecast),
            targets,
            np.ones_like(targets),
            history,
            np.ones_like(history),
            QUANTILES,
            ["D"] * 16,
        )
        return np.nanmean(out["mase"])

    # measured ~97x on seasonal data, so this is a sharp detector rather than
    # a marginal one; 10x leaves room for seed variation without going slack
    assert mase(shifted) > 10.0 * mase(aligned)


def test_ragged_history_is_honored_via_mask():
    """Left-padded short series are the point of the native mode, so a
    padded series must score identically to the same series unpadded."""
    rng = np.random.default_rng(3)
    short = rng.normal(size=(1, 12))
    targets = rng.normal(size=(1, 4))
    forecast = _quantile_stack(np.zeros_like(targets))

    padded_hist = np.concatenate([np.full((1, 8), 999.0), short], axis=1)
    padded_mask = np.concatenate([np.zeros((1, 8)), np.ones((1, 12))], axis=1)

    tight = accuracy.per_series_metrics(
        forecast,
        targets,
        np.ones_like(targets),
        short,
        np.ones_like(short),
        QUANTILES,
        ["D"],
    )
    padded = accuracy.per_series_metrics(
        forecast,
        targets,
        np.ones_like(targets),
        padded_hist,
        padded_mask,
        QUANTILES,
        ["D"],
    )
    for name in ("nmse", "mase"):
        assert padded[name][0] == pytest.approx(tight[name][0]), name


def test_constant_context_yields_nan_rather_than_a_huge_nmse():
    """A degenerate scale must drop out of the pooled mean instead of
    dominating it, which is how the training path treats it too."""
    targets = np.array([[1.0, 2.0]])
    history = np.full((1, 10), 5.0)
    out = accuracy.per_series_metrics(
        _quantile_stack(np.zeros_like(targets)),
        targets,
        np.ones_like(targets),
        history,
        np.ones_like(history),
        QUANTILES,
        ["D"],
    )
    assert np.isnan(out["nmse"][0])
    assert np.isnan(out["mase"][0])
    assert np.isfinite(out["mae"][0])


def test_mape_reports_coverage_when_actuals_are_zero():
    """Favorita is mostly zeros by Kaggle's convention, so MAPE there is
    computed over a small subset and the harness must say so."""
    targets = np.array([[0.0, 0.0, 0.0, 4.0]])
    history = np.arange(1.0, 21.0).reshape(1, 20)
    out = accuracy.per_series_metrics(
        _quantile_stack(np.array([[1.0, 1.0, 1.0, 2.0]])),
        targets,
        np.ones_like(targets),
        history,
        np.ones_like(history),
        QUANTILES,
        ["D"],
    )
    assert out["mape_coverage"][0] == pytest.approx(0.25)
    assert out["mape"][0] == pytest.approx(50.0)
    assert np.isfinite(out["smape"][0])


def test_nmse_and_mase_are_scale_free_but_mae_is_not():
    rng = np.random.default_rng(4)
    history = rng.normal(loc=10.0, scale=2.0, size=(5, 40))
    targets = rng.normal(loc=10.0, scale=2.0, size=(5, 6))
    forecast = _quantile_stack(targets + rng.normal(scale=0.5, size=targets.shape))
    args = (np.ones((5, 6)), history, np.ones_like(history), QUANTILES, ["D"] * 5)

    base = accuracy.per_series_metrics(forecast, targets, *args)
    scaled = accuracy.per_series_metrics(
        forecast * 50.0,
        targets * 50.0,
        np.ones((5, 6)),
        history * 50.0,
        np.ones_like(history),
        QUANTILES,
        ["D"] * 5,
    )
    assert np.allclose(base["nmse"], scaled["nmse"])
    assert np.allclose(base["mase"], scaled["mase"])
    assert np.allclose(base["wql"], scaled["wql"])
    assert not np.allclose(base["mae"], scaled["mae"])


def test_pool_drops_undefined_series_and_counts_them():
    per_series = {"mase": np.array([1.0, np.nan, 3.0])}
    pooled = accuracy.pool(per_series)
    assert pooled["mase"] == pytest.approx(2.0)
    assert pooled["mase_n"] == 2

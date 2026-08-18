"""The classical reference baselines.

The check that matters most is that statsforecast's `SeasonalNaive` agrees
with the harness's own `score.seasonal_naive`, because MASE is defined
against that rule. If the two ever diverge, every baseline MASE is measured
against a denominator its own numerator does not share.
"""

import numpy as np
import pytest

from src.eval import baselines, score
from src.eval.suites import EvalSeries


def _series(length, period, seed=0, subset="s"):
    rng = np.random.default_rng(seed)
    season = rng.normal(size=period)
    idx = np.arange(length)
    values = season[idx % period] + rng.normal(scale=0.05, size=length) + 10.0
    return EvalSeries(
        suite="t",
        subset=subset,
        item_id="i",
        history=values[:-6],
        actual=values[-6:],
        period=period,
        freq="M",
    )


def test_seasonal_naive_matches_the_harness_implementation():
    items = [_series(60, 12, seed=i) for i in range(8)]
    ours = score.seasonal_naive(items, 6, 1)[:, :, 0]
    theirs = baselines.forecast("seasonal_naive", items, 6)[
        :, :, baselines.QUANTILES.index(0.5)
    ]
    assert np.allclose(ours, theirs)


def test_quantiles_are_sorted_and_centred_on_the_median():
    out = baselines.forecast("ets", [_series(72, 12, seed=1)], 6)
    assert out.shape == (1, 6, len(baselines.QUANTILES))
    assert np.all(np.diff(out[0], axis=-1) >= 0)


@pytest.mark.parametrize("name", baselines.BASELINES)
def test_a_series_too_short_to_fit_yields_nan_rather_than_raising(name):
    """Short series are the norm in M1 and M3, so an estimator that cannot
    take one must be reported as unfitted, not crash the whole subset."""
    item = EvalSeries(
        suite="t",
        subset="s",
        item_id="i",
        history=np.array([1.0, 2.0]),
        actual=np.array([3.0, 4.0]),
        period=12,
        freq="Y",
    )
    out = baselines.forecast(name, [item], 2)
    assert out.shape == (1, 2, len(baselines.QUANTILES))
    assert np.isnan(out).all()


def test_an_unusable_period_falls_back_to_one():
    """Two rules travel together here. The estimated models cannot hold a
    period of 8,640, GIFT-Eval's `bizitobs_*` value, and no model can
    estimate a season it has seen fewer than twice. SeasonalNaive keeps any
    period, because MASE's denominator does."""
    long_period = _series(400, 8640, seed=2)
    assert baselines.usable_period("arima", long_period, 400) == 1
    assert baselines.usable_period("ets", long_period, 400) == 1

    fits = _series(400, 12, seed=3)
    assert baselines.usable_period("arima", fits, 400) == 12
    assert baselines.usable_period("seasonal_naive", fits, 400) == 12

    barely = _series(400, 20, seed=4)
    assert baselines.usable_period("ets", barely, 30) == 1

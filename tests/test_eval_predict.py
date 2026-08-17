"""Batching, padding, and scoring layers of the eval harness.

These use a stub Forecaster rather than a real checkpoint: what is under
test is that ragged suites survive the trip through a fixed-width model
interface without losing series, not any model's numerics.
"""

import numpy as np
import pytest

from src.eval import predict, score
from src.eval.protocol import Forecaster
from src.eval.suites import EvalSeries

QUANTILES = [0.1, 0.5, 0.9]


class ConstantForecaster:
    """Echoes the last observed context point across the horizon."""

    quantiles = QUANTILES
    context_length = 16
    horizon = 8

    def __init__(self):
        self.seen = []

    def predict(self, context, valid, freqs):
        self.seen.append(context.shape[0])
        last = context[:, -1]
        point = np.repeat(last[:, None], self.horizon, axis=1)
        return np.repeat(point[..., None], len(self.quantiles), axis=-1)


def _series(n, history_len, horizon=4, period=1, subset="s"):
    return [
        EvalSeries(
            suite="test",
            subset=subset,
            item_id=f"i{i}",
            history=np.arange(1.0, history_len + 1.0) + i,
            actual=np.full(horizon, float(i)),
            period=period,
            freq="D",
        )
        for i in range(n)
    ]


def test_stub_forecaster_satisfies_the_protocol():
    assert isinstance(ConstantForecaster(), Forecaster)


def test_short_history_is_left_padded_and_masked_not_dropped():
    """The suites are mostly shorter than the context: 100% of M1/M3/Tourism
    and 95% of M4. A length filter here would empty them."""
    series = _series(3, history_len=5)
    context, valid = predict.build_context(series, 16)

    assert context.shape == (3, 16)
    assert valid[:, :11].sum() == 0  # padding is masked out
    assert valid[:, 11:].all()
    assert context[0, -1] == 5.0  # the series ends at the right edge


def test_long_history_is_truncated_to_the_most_recent_context():
    series = _series(1, history_len=50)
    context, valid = predict.build_context(series, 16)
    assert valid.all()
    assert context[0, -1] == 50.0
    assert context[0, 0] == 35.0


def test_every_series_is_forecast_even_across_batches():
    series = _series(70, history_len=5)
    forecaster = ConstantForecaster()
    out = predict.run(forecaster, series, batch_size=32)

    assert out.values.shape == (70, 4, len(QUANTILES))
    assert forecaster.seen == [32, 32, 6]
    assert len(out.item_ids) == 70


def test_mixed_horizons_in_one_call_are_rejected():
    series = _series(2, history_len=5, horizon=4) + _series(2, history_len=5, horizon=6)
    with pytest.raises(ValueError, match="mixed horizons"):
        predict.run(ConstantForecaster(), series)


def test_horizon_beyond_the_native_window_is_rejected():
    """No rollout is implemented, so a suite that outgrows the model must
    fail loudly rather than be silently truncated to a shorter horizon."""
    series = _series(2, history_len=5, horizon=12)
    with pytest.raises(ValueError, match="autoregressive rollout"):
        predict.run(ConstantForecaster(), series)


def test_seasonal_naive_repeats_the_last_season():
    series = _series(1, history_len=12, horizon=6, period=4)
    baseline = score.seasonal_naive(series, 6, len(QUANTILES))

    # history is 1..12, so the last season is 9,10,11,12 and it cycles
    assert np.allclose(baseline[0, :, 0], [9, 10, 11, 12, 9, 10])


def test_seasonal_naive_falls_back_to_lag_one_when_the_season_does_not_fit():
    """Matches accuracy.seasonal_naive_mae's fallback, so the MASE numerator
    and denominator always use the same rule."""
    series = _series(1, history_len=3, horizon=4, period=12)
    baseline = score.seasonal_naive(series, 4, len(QUANTILES))
    assert np.allclose(baseline[0, :, 0], [3, 3, 3, 3])


def test_summarize_breaks_down_by_subset():
    """Pooled means hide the imbalance: one GIFT-Eval config is 45,000 of its
    319,209 instances."""
    series = _series(4, history_len=8, subset="a") + _series(
        2, history_len=8, subset="b"
    )
    out = predict.run(ConstantForecaster(), series)
    per_series = score.score(out)
    pooled, by_subset = score.summarize(out, per_series)

    assert set(by_subset) == {"a", "b"}
    assert by_subset["a"]["mae_n"] == 4
    assert by_subset["b"]["mae_n"] == 2
    assert pooled["mae_n"] == 6


def test_rolling_windows_are_ordered_oldest_first():
    """stability.excess_volatility reads its before/update pairing off this
    ordering, so a reversed window axis silently inverts EV's sign."""
    series = [
        EvalSeries(
            suite="t",
            subset="s",
            item_id="i0",
            history=np.arange(0.0, 20.0),
            actual=np.arange(20.0, 24.0),
            period=1,
            freq="D",
        )
    ]
    out = predict.run_rolling(
        ConstantForecaster(), series, horizon=4, stride=2, n_windows=3
    )

    assert out.values.shape == (1, 3, 4, 1, len(QUANTILES))
    assert out.actual.shape == (1, 3, 4, 1)
    # the last window ends at the series' final point, earlier ones step back
    assert np.allclose(out.actual[0, 2, :, 0], [20, 21, 22, 23])
    assert np.allclose(out.actual[0, 1, :, 0], [18, 19, 20, 21])
    assert np.allclose(out.actual[0, 0, :, 0], [16, 17, 18, 19])


def test_rolling_rejects_a_stride_that_does_not_overlap():
    series = _series(1, history_len=40, horizon=4)
    with pytest.raises(ValueError, match="do not overlap"):
        predict.run_rolling(
            ConstantForecaster(), series, horizon=4, stride=4, n_windows=3
        )


def test_rolling_rejects_series_too_short_rather_than_dropping_them():
    series = _series(1, history_len=6, horizon=4)
    with pytest.raises(ValueError, match="lower n_windows rather than dropping"):
        predict.run_rolling(
            ConstantForecaster(), series, horizon=4, stride=2, n_windows=8
        )


def test_stability_scores_run_end_to_end():
    series = _series(6, history_len=40, horizon=4, period=1)
    out = predict.run_rolling(
        ConstantForecaster(), series, horizon=4, stride=1, n_windows=5
    )
    scores = score.score_stability(out)

    assert set(scores) == {"excess_volatility", "sfpc"}
    assert np.isfinite(scores["excess_volatility"])
    # a persistence forecaster never revises a given date's forecast much
    assert scores["sfpc"] >= 0.0

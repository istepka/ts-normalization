"""Scores forecasts and breaks the result down by suite subset.

The seasonal-naive baseline lives here rather than behind the `Forecaster`
protocol because it is not a checkpoint. Keeping it a plain function also
lets it read each series' declared seasonal period, which the protocol does
not carry: M4 scores on its own competition seasonality while GIFT-Eval uses
the gluonts convention, so a baseline deriving the period from the frequency
string would disagree with the MASE denominator on exactly those suites.
"""

import numpy as np

from src.eval.predict import Forecasts, RollingForecasts
from src.eval.suites import EvalSeries
from src.metrics import accuracy, stability


def seasonal_naive(
    series: list[EvalSeries], horizon: int, n_quantiles: int
) -> np.ndarray:
    """Repeats each series' last full season forward, as [N, H, Q].

    The MASE denominator is the in-sample error of this same rule, so this
    forecast scores MASE near 1 by construction. That makes it the harness's
    calibration check: a horizon misaligned by one step sends it to roughly
    100 on seasonal data rather than to something merely a bit worse.

    Reads each series' full history rather than the model's padded context,
    because a baseline is not limited by any model's window. GIFT-Eval's
    `bizitobs_*` configs have a seasonal period of 8,640 against a 512-point
    model context, so reading the context would silently collapse them to a
    random walk and make the comparison against published numbers meaningless.

    Where the period does not fit in a series' history the lag falls back to
    1, matching `accuracy.seasonal_naive_mae`, so the MASE numerator and
    denominator always use the same rule.
    """
    out = np.empty((len(series), horizon, n_quantiles))
    steps = np.arange(horizon)
    for i, item in enumerate(series):
        history = item.history
        period = item.period if item.period < len(history) else 1
        period = max(period, 1)
        out[i] = np.repeat(
            history[len(history) - period + (steps % period)][:, None],
            n_quantiles,
            axis=1,
        )
    return out


def score(
    forecasts: Forecasts, series: list[EvalSeries] | None = None
) -> dict[str, np.ndarray]:
    """Every accuracy metric, per series, in the order the suite gave.

    Pass `series` so the MASE denominator is computed over each series' full
    history rather than the model's padded context. Without it the
    denominator is bounded by `context_length`, which is a model constraint
    leaking into a metric definition and moves MASE by up to 2x on long
    series (see the GIFT-Eval baseline comparison in
    notes/agentic_logs/2026-08-16-eval-harness.md).
    """
    naive_mae = None
    if series is not None:
        naive_mae = accuracy.ragged_seasonal_naive_mae(
            [item.history for item in series], forecasts.periods
        )
    return accuracy.per_series_metrics(
        forecasts.values,
        forecasts.actual,
        forecasts.actual_mask,
        forecasts.history,
        forecasts.history_mask,
        forecasts.quantiles,
        forecasts.periods,
        naive_mae=naive_mae,
    )


def score_stability(rolling: RollingForecasts) -> dict[str, float]:
    """Excess Volatility and symmetric Forecast Percentage Change.

    Both are single numbers over the whole batch rather than per series,
    since each aggregates across the window pairs it forms. sFPC reads the
    median forecast, being a point-forecast measure.
    """
    median = rolling.values[..., rolling.quantiles.index(0.5)]
    return {
        "excess_volatility": stability.excess_volatility(
            rolling.actual,
            rolling.values,
            rolling.quantiles,
            stride=rolling.stride,
            mask=rolling.mask,
        ),
        "sfpc": stability.forecast_percentage_change(
            median, stride=rolling.stride, mask=rolling.mask
        ),
    }


def summarize(
    forecasts: Forecasts, per_series: dict[str, np.ndarray]
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Pools over the whole suite and again per subset.

    Both are reported because the suites are wildly unbalanced. GIFT-Eval's
    `bitbrains_fast_storage/5T` alone is 45,000 of its 319,209 instances, so
    a pooled mean is close to a report on that one config unless the
    per-subset table sits next to it.
    """
    pooled = accuracy.pool(per_series)
    subsets = np.asarray(forecasts.subsets)
    by_subset = {}
    for name in sorted(set(forecasts.subsets)):
        rows = subsets == name
        by_subset[name] = accuracy.pool({k: v[rows] for k, v in per_series.items()})
    return pooled, by_subset

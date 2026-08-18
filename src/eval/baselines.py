"""Classical reference forecasts over the held-out suites.

These are not zero-shot in the sense the TSFMs are. Each one is estimated on
the series it forecasts, at forecast time, from that series' own history.
They belong in the results as a reference column, clearly separated from the
pretrained models, not as a competitor in the same protocol.

`statsforecast`'s `AutoETS` and `AutoARIMA` search over model forms per
series. That search is a per-series fit of the model *structure*, so the
model specification here is fixed up front and only its parameters are
estimated. `AutoETS(model="AAA")` is the fixed-form variant of the same
estimator, not the search.

Forecasts come back as `[N, H, Q]` on the same nine quantiles the harness
scores, so `src/eval/score.py` scores a baseline through exactly the code
that scores a checkpoint.
"""

import numpy as np
from statsforecast.models import ARIMA, AutoETS, SeasonalNaive

from src.eval.suites import EvalSeries

# The nine quantiles the harness reports, and the statsforecast prediction
# interval levels that produce them. A level L gives the (1-L/100)/2 and
# 1-(1-L/100)/2 quantiles as its `lo-L` and `hi-L` bounds.
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
LEVELS = [20, 40, 60, 80]
BASELINES = ("seasonal_naive", "ets", "arima")

# Non-seasonal ARIMA(1,1,1) with a seasonal (0,1,1) when the series has a
# usable period, which is the airline model's seasonal part. Chosen once for
# every series rather than searched, which is the whole point.
ARIMA_ORDER = (1, 1, 1)
ARIMA_SEASONAL_ORDER = (0, 1, 1)
# ETS needs at least two full seasons to estimate a seasonal component, and
# more than that to be worth trusting.
MIN_SEASONS = 2
# Above this the seasonal state is not estimable in practice. R's
# forecast::ets refuses a period over 24 for the same reason, and a seasonal
# ARIMA at GIFT-Eval's `bizitobs_*` period of 8,640 would carry 8,640 seasonal
# states per series. Those configs fall back to a non-seasonal fit here.
# SeasonalNaive is exempt: it is a lag lookup at any period, and it must keep
# the suite's declared period to stay consistent with the MASE denominator.
MAX_SEASON = 24


def build_model(name: str, season_length: int):
    """The fixed-form estimator for one series' seasonal period.

    `season_length` of 1 means the series has no usable period, so the
    seasonal component is dropped rather than estimated on nothing.
    """
    if name == "seasonal_naive":
        return SeasonalNaive(season_length=season_length)
    if name == "ets":
        # additive error, additive trend, additive season. Additive rather
        # than multiplicative because Favorita and several GIFT-Eval configs
        # hold zeros and negatives, which multiplicative error cannot take.
        model = "AAA" if season_length > 1 else "AAN"
        return AutoETS(season_length=season_length, model=model)
    if name == "arima":
        seasonal = ARIMA_SEASONAL_ORDER if season_length > 1 else (0, 0, 0)
        return ARIMA(
            order=ARIMA_ORDER,
            season_length=season_length,
            seasonal_order=seasonal,
        )
    raise ValueError(f"unknown baseline {name!r}, expected one of {BASELINES}")


def usable_period(name: str, item: EvalSeries, observed: int) -> int:
    """The seasonal period this series can actually support.

    Falls back to 1 where the period does not fit, matching
    `accuracy.seasonal_naive_mae` so the MASE numerator and denominator
    always describe the same rule. The estimated models additionally need
    several full seasons before a seasonal term is worth carrying, and a
    period they can actually hold state for.
    """
    period = int(item.period)
    if period < 2 or observed < MIN_SEASONS * period:
        return 1
    if name != "seasonal_naive" and period > MAX_SEASON:
        return 1
    return period


def forecast_series(name: str, item: EvalSeries, horizon: int) -> np.ndarray:
    """One series' quantile forecast, `[H, Q]`, or all-NaN if it fails.

    A real corpus produces series a fixed-form estimator cannot fit: too
    short, constant, or singular under the differencing the order implies.
    Those come back as NaN and are dropped by `accuracy.pool`, which reports
    the surviving count per metric, rather than being replaced by a fallback
    forecast that would quietly become a different model.
    """
    history = np.asarray(item.history, dtype=np.float64)
    history = history[np.isfinite(history)]
    out = np.full((horizon, len(QUANTILES)), np.nan)
    if len(history) < 3:
        return out

    model = build_model(name, usable_period(name, item, len(history)))
    try:
        result = model.forecast(y=history, h=horizon, level=LEVELS)
    except (ValueError, ZeroDivisionError, RuntimeError, np.linalg.LinAlgError):
        # Real corpora produce series a fixed-form estimator cannot take:
        # constant, singular under the implied differencing, or too short for
        # the seasonal state. Anything outside this set is a bug and should
        # surface rather than be scored as a missing forecast.
        return out

    out[:, QUANTILES.index(0.5)] = np.asarray(result["mean"], dtype=np.float64)
    for level in LEVELS:
        lower = (1.0 - level / 100.0) / 2.0
        out[:, QUANTILES.index(round(lower, 1))] = np.asarray(
            result[f"lo-{level}"], dtype=np.float64
        )
        out[:, QUANTILES.index(round(1.0 - lower, 1))] = np.asarray(
            result[f"hi-{level}"], dtype=np.float64
        )
    # The intervals are symmetric around the mean but the solver can still
    # emit a crossing pair on a degenerate fit, which would make the pinball
    # loss meaningless. Sorting restores monotonicity across quantiles.
    return np.sort(out, axis=1)


def forecast(name: str, series: list[EvalSeries], horizon: int) -> np.ndarray:
    """`[N, H, Q]` over a whole subset, in the order given."""
    return np.stack([forecast_series(name, item, horizon) for item in series])

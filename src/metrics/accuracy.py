"""Per-series accuracy metrics for the evaluation harness.

Every metric here takes a ragged batch, meaning left-padded history plus a
validity mask, because the eval suites forecast whatever history a series
actually has rather than a fixed 512 points. All are computed per series and
returned unaggregated so callers can pool, take medians, or break down by
suite without recomputing.

nMSE and MASE follow the definitions already used in training and in
`src/metrics/scale_free.py`, so `fixed`-mode numbers are directly comparable
to training-time eval:
  - nMSE is the squared error divided by the context variance, which is the
    same as MSE in the models' normalized space.
  - MASE is the horizon MAE over the in-sample seasonal-naive MAE.
CRPS and WQL come from `src/metrics/eval_losses.py` in the doubled
sum-over-quantiles convention.
"""

import numpy as np

from src.metrics.eval_losses import quantile_loss

# A context whose spread sits at or below this has no usable scale, so the
# scale-relative metrics are NaN there rather than dividing by ~0 and
# dominating any mean. Matches the clamp floor used by the model adapters.
SIGMA_FLOOR = 1e-8


def context_scale(
    history: np.ndarray, history_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-series mean and standard deviation over the valid history.

    Returns `(mu, sigma)`, each `[N]`. `sigma` is NaN where the context is
    constant or empty, which propagates into nMSE rather than silently
    producing a huge value.
    """
    counts = history_mask.sum(axis=1)
    mu = np.where(
        counts > 0, (history * history_mask).sum(axis=1) / np.maximum(counts, 1), np.nan
    )
    centered = (history - mu[:, None]) * history_mask
    var = np.where(
        counts > 1, (centered**2).sum(axis=1) / np.maximum(counts - 1, 1), np.nan
    )
    sigma = np.sqrt(var)
    return mu, np.where(sigma > SIGMA_FLOOR, sigma, np.nan)


def seasonal_naive_mae(
    history: np.ndarray, history_mask: np.ndarray, periods: np.ndarray
) -> np.ndarray:
    """In-sample seasonal-naive MAE per series, the standard MASE denominator.

    `periods` is the seasonal lag m per series. Where m does not fit in a
    series' valid history the lag falls back to 1, the plain random-walk
    naive baseline, matching `src/metrics/forecast.py`. Dropping those series
    instead would silently remove the shortest ones from every suite, which
    is exactly the filtering the harness exists to avoid.

    A pair contributes only where both endpoints are valid. Series whose
    history is exactly constant yield ~0 and are returned as NaN.
    """
    length = history.shape[1]
    valid_counts = history_mask.sum(axis=1)
    lag = np.where(periods >= np.maximum(valid_counts, 1), 1, np.maximum(periods, 1))

    idx = np.arange(length)[None, :]
    src = idx - lag[:, None]
    gather = np.clip(src, 0, None)
    lagged = np.take_along_axis(history, gather, axis=1)
    lagged_mask = np.take_along_axis(history_mask, gather, axis=1)

    pair = history_mask * lagged_mask * (src >= 0)
    total = (np.abs(history - lagged) * pair).sum(axis=1)
    denom = pair.sum(axis=1)
    mae = np.where(denom > 0, total / np.maximum(denom, 1), np.nan)
    return np.where(mae > SIGMA_FLOOR, mae, np.nan)


def per_series_metrics(
    forecasts: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    history: np.ndarray,
    history_mask: np.ndarray,
    quantiles: list[float],
    periods: np.ndarray,
) -> dict[str, np.ndarray]:
    """Every accuracy metric, per series.

    `forecasts`: [N, H, Q] quantile forecasts, quantile axis last, and
    `quantiles` must contain 0.5 since the point metrics use the median.
    `targets`/`target_mask`: [N, H]. `history`/`history_mask`: [N, L],
    left-padded. `periods`: [N] MASE seasonal lag per series.

    The seasonal period is supplied rather than derived from a frequency
    string because it is part of the suite's protocol, and not every suite
    has a frequency to derive it from. M3 Other carries neither a frequency
    nor a start timestamp, so its period is declared as 1 by the loader.

    Returns arrays of shape [N]. NaN marks a series the metric is undefined
    for (constant context for nMSE, constant history for MASE, zero actuals
    for MAPE) so callers drop it with nanmean rather than having a degenerate
    denominator dominate the pooled number.
    """
    if forecasts.shape[:2] != targets.shape:
        raise ValueError(
            f"forecasts {forecasts.shape} and targets {targets.shape} disagree"
        )
    if len(periods) != targets.shape[0]:
        raise ValueError(f"{len(periods)} periods for {targets.shape[0]} series")

    median = forecasts[..., quantiles.index(0.5)]
    counts = np.maximum(target_mask.sum(axis=1), 1)
    error = (median - targets) * target_mask

    mae = np.abs(error).sum(axis=1) / counts
    mse = (error**2).sum(axis=1) / counts

    _, sigma = context_scale(history, history_mask)
    naive = seasonal_naive_mae(history, history_mask, np.asarray(periods))

    # MAPE is undefined at zero actuals. Favorita fills 20.5% of points to
    # zero by Kaggle's convention, so those positions are excluded per series
    # and the surviving fraction is reported as coverage. A MAPE over a
    # handful of points is not comparable to one over a full horizon.
    nonzero = target_mask * (np.abs(targets) > 0)
    ape = np.where(nonzero > 0, np.abs(error) / np.maximum(np.abs(targets), 1e-12), 0.0)
    ape_counts = nonzero.sum(axis=1)
    mape = np.where(
        ape_counts > 0, 100.0 * ape.sum(axis=1) / np.maximum(ape_counts, 1), np.nan
    )

    # sMAPE has a bounded denominator, so it survives the zeros MAPE cannot.
    smape_den = np.abs(median) + np.abs(targets)
    smape_terms = np.where(
        smape_den > 0, 2.0 * np.abs(error) / np.maximum(smape_den, 1e-12), 0.0
    )
    smape = 100.0 * (smape_terms * target_mask).sum(axis=1) / counts

    crps = np.array(
        [
            quantile_loss(
                forecasts[i : i + 1],
                targets[i : i + 1],
                quantiles,
                target_mask[i : i + 1],
                aggregate="mean",
            )
            for i in range(targets.shape[0])
        ]
    )
    abs_target = (np.abs(targets) * target_mask).sum(axis=1)
    wql_total = quantile_loss(
        forecasts, targets, quantiles, target_mask, aggregate=None
    ).sum(axis=(1, 2)) / len(quantiles)
    wql = np.where(abs_target > 0, wql_total / np.maximum(abs_target, 1e-12), np.nan)

    return {
        "mae": mae,
        "mse": mse,
        "nmse": mse / sigma**2,
        "mase": mae / naive,
        "mape": mape,
        "smape": smape,
        "crps": crps,
        "wql": wql,
        "mape_coverage": ape_counts / counts,
    }


def pool(per_series: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean over series, ignoring the NaN a metric marks as undefined.

    Reported alongside a count per metric, because a mean over 200 of 83,207
    Favorita series is a different claim than a mean over all of them and the
    difference is invisible in the number itself.
    """
    out = {}
    for name, values in per_series.items():
        finite = np.isfinite(values)
        out[name] = float(np.nanmean(values[finite])) if finite.any() else float("nan")
        out[f"{name}_n"] = int(finite.sum())
    return out

"""Forecast stability metrics over overlapping rolling windows.

These measure how much a model revises its forecast of a fixed target date
as the forecast creation date advances, which accuracy metrics cannot see.
Both require overlapping windows and raise when `stride >= H`, so they run
only in the harness's `rolling` mode. Every official protocol in the six
eval suites is single-window or exactly non-overlapping (GIFT-Eval rolls
with `distance=prediction_length`), so neither metric is defined there. See
notes/agentic_logs/2026-08-16-eval-harness.md.
"""

import numpy as np

from src.metrics.eval_losses import quantile_loss


def _check_stride(stride: int, horizon: int) -> None:
    if stride == horizon:
        raise ValueError(
            f"stride={stride} equals H={horizon}: windows are non-overlapping "
            "so each target date has exactly one forecast, and forecast "
            "stability is undefined"
        )
    if stride > horizon:
        raise ValueError(
            f"stride={stride} > H={horizon}: some target dates have no "
            "forecast coverage; review the stride used in this experiment"
        )


def reshape_windows_by_date(
    x: np.ndarray, stride: int, mask: np.ndarray | None = None
) -> np.ndarray:
    """Rearranges overlapping forecast windows [B, T, H, C] into
    [B, (T-1)*stride + H, H, C], grouping predictions by target date.

    In a rolling setup several windows land on the same target date, e.g.
    the 1-step-ahead prediction from window t and the 2-step-ahead from
    window t-1 both target date t. This collects those into one row so
    forecasts of the same date can be compared directly.

    The output has one row per unique target date and H columns, one per
    horizon that could reach that date. Edge dates are partially covered and
    hold NaN for horizons that do not reach them. Masked positions
    (`mask == 0`) become NaN before rearranging.
    """
    batch, n_windows, horizon, channels = x.shape
    n_dates = (n_windows - 1) * stride + horizon

    if mask is not None:
        x = np.where(mask == 0, np.nan, x)

    t_grid, h_grid = np.meshgrid(
        np.arange(n_windows), np.arange(horizon), indexing="ij"
    )
    date_grid = t_grid * stride + h_grid
    out = np.full((batch, n_dates, horizon, channels), np.nan)
    out[:, date_grid, h_grid, :] = x
    return out


def excess_volatility(
    targets: np.ndarray,
    preds: np.ndarray,
    quantiles: list[float],
    stride: int = 1,
    scaling: bool = True,
    mask: np.ndarray | None = None,
) -> float:
    """Excess Volatility: the cost of a forecast revision net of the accuracy
    it bought.

        EV = QL(y_update_median, y_before)
           - (QL(y, y_before) - QL(y, y_update))

    For each overlapping pair of windows predicting the same target date, the
    revision cost is how badly the older forecast predicts the newer one, and
    the subtracted term is how much accuracy the revision actually gained. A
    positive EV means the model churned more than the improvement justified.

    `targets`: [B, T, H, C]. `preds`: [B, T, H, C, Q], quantile axis last.
    The window axis T must be ordered oldest-first, window t created before
    window t+1, which is what `src/eval/predict.py` emits. `quantiles` must
    contain 0.5. `mask`: [B, T, H, C], 1 = real timestep.
    With `scaling`, EV is divided by the total absolute target so it is
    comparable across series of different magnitude.

    Lower is better. A model whose forecasts converge on the truth as the
    creation date approaches scores near zero; one that thrashes without
    gaining accuracy scores higher. `forecast_percentage_change` measures the
    churn alone and is unaffected by the pairing order, since it is symmetric
    in the two forecasts.
    """
    batch, n_windows, horizon, channels, n_quantiles = preds.shape
    n_dates = (n_windows - 1) * stride + horizon
    _check_stride(stride, horizon)

    reshaped_preds = reshape_windows_by_date(
        preds.reshape(batch, n_windows, horizon, channels * n_quantiles),
        stride,
        mask=np.repeat(mask, n_quantiles, axis=-1) if mask is not None else None,
    )
    reshaped_y = reshape_windows_by_date(targets, stride)
    reshaped_mask = reshape_windows_by_date(mask, stride) if mask is not None else None

    # This harness stacks the window axis oldest-first, so window t was
    # created before t+1. For a fixed target date d the source window is
    # t = (d - h) / stride, so consecutive creation dates sit `stride` apart
    # in h, and ascending h walks BACKWARDS in creation date: h + stride is
    # the older forecast, h the newer one.
    #
    # Stepping by 1 instead of by stride is only correct when stride == 1.
    # For any larger stride the intermediate h slots are structurally empty
    # (t would not be an integer), so every pair mixes a real forecast with a
    # NaN and the metrics collapse to EV = 0 and sFPC = NaN.
    #
    # The direction has to be read off the window ordering rather than
    # assumed. Under a newest-first convention the slices swap. Getting it
    # wrong flips the sign of the accuracy term, which makes EV reward churn
    # and penalize convergence, and does so quietly: both arms land near the
    # same inflated value rather than looking obviously broken.
    older = reshaped_preds[:, :, stride:, :]
    newer = reshaped_preds[:, :, :-stride, :]

    # Uncovered (date, h) slots are NaN by construction, and a pair is usable
    # only where both endpoints exist. Deriving validity from the coverage
    # pattern keeps edge dates from contributing zero-filled terms when no
    # caller mask is supplied.
    n_pairs = horizon - stride
    pair_valid = (np.isfinite(older) & np.isfinite(newer)).astype(float)
    pair_valid = pair_valid.reshape(batch, n_dates, n_pairs, channels, n_quantiles)
    pair_mask = pair_valid[..., 0]
    if reshaped_mask is not None:
        pair_mask = pair_mask * np.logical_and(
            np.nan_to_num(reshaped_mask[:, :, stride:, :], nan=0.0),
            np.nan_to_num(reshaped_mask[:, :, :-stride, :], nan=0.0),
        ).astype(float)
    pair_mask = pair_mask * np.isfinite(reshaped_y[:, :, stride:, :]).astype(float)

    before = np.nan_to_num(older, nan=0.0).reshape(
        batch, n_dates, n_pairs, channels, n_quantiles
    )
    update = np.nan_to_num(newer, nan=0.0).reshape(
        batch, n_dates, n_pairs, channels, n_quantiles
    )
    reshaped_y = np.nan_to_num(reshaped_y[:, :, stride:, :], nan=0.0)

    mid = quantiles.index(0.5)
    update_median = update[..., mid]

    # aggregate=None keeps the loss elementwise so the three terms compose
    # before any sum; dividing by Q undoes this module's sum over quantiles.
    revision_cost = (
        quantile_loss(before, update_median, quantiles, pair_mask, None) / n_quantiles
    )
    accuracy_before = (
        quantile_loss(before, reshaped_y, quantiles, pair_mask, None) / n_quantiles
    )
    accuracy_update = (
        quantile_loss(update, reshaped_y, quantiles, pair_mask, None) / n_quantiles
    )

    ev = (revision_cost - (accuracy_before - accuracy_update)).sum()
    if not scaling:
        return float(ev)
    return float(ev / ((np.abs(reshaped_y) * pair_mask).sum() + 1e-8))


def forecast_percentage_change(
    preds: np.ndarray,
    stride: int = 1,
    scaling: bool = True,
    mask: np.ndarray | None = None,
) -> float:
    """Symmetric Forecast Percentage Change: the relative size of forecast
    revisions between consecutive forecast creation dates.

        sFPC = 200 * mean(|y_update - y_before| / (|y_update| + |y_before|))

    Unlike EV this ignores the ground truth entirely, so it measures churn
    alone. Higher means less stable. With `scaling=False` the denominator is
    dropped and this is the mean absolute revision, still scaled by 200.

    `preds`: [B, T, H, C] point forecasts. `mask`: [B, T, H, C], 1 = real
    timestep, masked positions drop out of the mean via NaN.
    """
    horizon = preds.shape[2]
    _check_stride(stride, horizon)

    # Paired stride apart, not 1 apart: see excess_volatility's comment on
    # why adjacent h slots are structurally empty for stride > 1.
    reshaped = reshape_windows_by_date(preds, stride, mask=mask)
    before = reshaped[:, :, stride:, :]
    update = reshaped[:, :, :-stride, :]

    num = np.abs(update - before)
    den = np.abs(update) + np.abs(before) + 1e-8
    return float(200.0 * np.nanmean(num / den if scaling else num))

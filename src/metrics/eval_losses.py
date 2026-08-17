"""Numpy quantile loss for the evaluation harness.

`src/losses/quantile.py` holds the torch training objectives and stays as it
is. This module exists because the eval harness scores saved forecasts
without autograd, and because the stability metrics in
`src/metrics/stability.py` need the loss *before* any aggregation, which the
training forms never expose (they reduce to per-example at the least).

Convention: this is the doubled sum-over-quantiles CRPS form, matching
`crps_quantile_loss` and the GIFT-Eval leaderboard, not the
mean-over-quantiles `pinball_loss`. The two differ by a factor of 2Q. The
harness uses this one form for every model so the CRPS column is comparable
across models that trained on different objectives (see
notes/agentic_logs/2026-08-16-eval-harness.md).
"""

import numpy as np


def quantile_loss(
    preds: np.ndarray,
    targets: np.ndarray,
    quantiles: list[float],
    mask: np.ndarray | None = None,
    aggregate: str | None = "sum",
) -> np.ndarray | float:
    """Doubled pinball loss, summed over the quantile axis.

    `preds`: [..., Q] with the quantile axis last. `targets`: [...], matching
    `preds` without its quantile axis. `mask`, when given, has `targets`'
    shape and zeroes out padded or missing positions (1 = real timestep).

    `aggregate` is one of:
      - None: returns the per-element loss, shape `[..., Q]`, already
        masked. This is what `excess_volatility` needs, since EV composes
        three losses elementwise before summing.
      - "sum": scalar sum over every axis.
      - "mean": sum over the quantile axis, then mean over valid positions.
        With no mask this is the plain mean over all positions.
    """
    q = np.asarray(quantiles, dtype=preds.dtype).reshape(*([1] * (preds.ndim - 1)), -1)
    diff = targets[..., None] - preds
    loss = 2.0 * np.abs(diff * ((diff <= 0.0).astype(preds.dtype) - q))
    if mask is not None:
        loss = loss * mask[..., None]

    if aggregate is None:
        return loss
    if aggregate == "sum":
        return float(loss.sum())
    if aggregate == "mean":
        per_position = loss.sum(axis=-1)
        if mask is None:
            return float(per_position.mean())
        return float(per_position.sum() / max(float(mask.sum()), 1.0))
    raise ValueError(f"unknown aggregate {aggregate!r}")


def weighted_quantile_loss(
    preds: np.ndarray,
    targets: np.ndarray,
    quantiles: list[float],
    mask: np.ndarray | None = None,
) -> float:
    """WQL: total quantile loss normalized by the total absolute target.

    This is the scale-free form the TSFM literature reports, and it is the
    reason WQL and CRPS are separate columns rather than one. CRPS is in the
    series' own units, WQL is not.
    """
    total = quantile_loss(preds, targets, quantiles, mask, aggregate="sum")
    denom = np.abs(targets) if mask is None else np.abs(targets) * mask
    return float(total / len(quantiles) / max(float(denom.sum()), 1e-8))

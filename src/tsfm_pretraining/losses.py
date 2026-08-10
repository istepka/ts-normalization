"""Loss-space primitives and dispersion/equity metrics shared by both TSFM stages.

Both the MOMENT and TimesFM adapters call into this module rather than each
re-deriving MSE, pinball loss, or the Gini/exposure bookkeeping, so the two
stages report metrics on exactly the same definitions (see
notes/05-timesfm-pretraining-loss-space-plan.md, "Dispersion and equity
metrics").
"""

import numpy as np
import torch
from pandas.tseries.frequencies import to_offset

TROUGH_STEP_CUTOFF = 2000  # matches the synthetic loss-space Gini table convention


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Elementwise squared error restricted to `mask` (1 = included).

    `reduction='none'` returns per-example mean squared error over the masked
    positions (shape `[batch]`), used for per-source dispersion metrics.
    """
    if mask.ndim != pred.ndim:
        raise ValueError(
            f"mask.ndim ({mask.ndim}) must equal pred.ndim ({pred.ndim}); a mask "
            "missing a channel/trailing dim silently broadcasts into a wrong-shaped "
            "cross product instead of a per-example mask -- unsqueeze it explicitly"
        )
    se = (pred - target) ** 2 * mask
    denom = mask.sum(dim=tuple(range(1, mask.ndim))).clamp_min(1.0)
    per_example = se.sum(dim=tuple(range(1, se.ndim))) / denom
    if reduction == "none":
        return per_example
    if reduction == "mean":
        return per_example.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Elementwise absolute error restricted to `mask` (1 = included).

    Same masking contract as `masked_mse`. Used as the MASE numerator.
    """
    if mask.ndim != pred.ndim:
        raise ValueError(
            f"mask.ndim ({mask.ndim}) must equal pred.ndim ({pred.ndim}); a mask "
            "missing a channel/trailing dim silently broadcasts into a wrong-shaped "
            "cross product instead of a per-example mask -- unsqueeze it explicitly"
        )
    ae = (pred - target).abs() * mask
    denom = mask.sum(dim=tuple(range(1, mask.ndim))).clamp_min(1.0)
    per_example = ae.sum(dim=tuple(range(1, ae.ndim))) / denom
    if reduction == "none":
        return per_example
    if reduction == "mean":
        return per_example.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


# GiftEvalPretrain stores gluonts/old-pandas frequency aliases (e.g. "M",
# "5T", "A-DEC") that newer pandas rejects in favor of "ME"/"min"/"YE-DEC".
_OLD_TO_NEW_FREQ_BASE = {
    "Y": "YE",
    "A": "YE",
    "Q": "QE",
    "M": "ME",
    "H": "h",
    "T": "min",
    "S": "s",
    "U": "us",
}


def parse_offset(freq: str):
    """Parses a GiftEvalPretrain frequency alias into a pandas offset,
    translating legacy aliases and preserving any multiplier and anchor
    (e.g. "5T" -> 5 minutes, "W-SUN" -> weekly)."""
    base, _, anchor = freq.partition("-")
    split = next((i for i, c in enumerate(base) if not c.isdigit()), len(base))
    mult, code = base[:split], base[split:]
    new_code = _OLD_TO_NEW_FREQ_BASE.get(code, code)
    anchor_suffix = f"-{anchor}" if anchor else ""
    return to_offset(f"{mult}{new_code}{anchor_suffix}")


def seasonal_period(freq: str) -> int:
    """Number of steps in the dominant seasonal cycle for a frequency, the
    MASE denominator's lag.

    Derived from the offset's actual duration rather than a lookup table of
    literal alias strings, so multipliers and anchors (e.g. "4S", "30T",
    "W-SUN", "A-DEC") are handled without enumerating every spelling. Follows
    the GIFT-Eval / gluonts `get_seasonality` convention: sub-daily
    frequencies take the daily cycle, daily takes the weekly cycle, weekly and
    yearly have no shorter cycle (1), monthly takes 12 and quarterly 4.
    """
    offset = parse_offset(freq)
    try:
        seconds = offset.nanos / 1e9
    except ValueError:
        # Non-fixed durations (weeks, months, quarters, years).
        name = type(offset).__name__
        if name == "Week":
            return 1
        if name.startswith("Month"):
            return 12
        if name.startswith("Quarter"):
            return 4
        return 1  # yearly and coarser
    if seconds < 86400:
        return round(86400 / seconds)  # sub-daily -> daily cycle
    if seconds == 86400:
        return 7  # daily -> weekly cycle
    return 1


def seasonal_naive_mae(
    context: torch.Tensor,
    valid: torch.Tensor,
    periods: torch.Tensor,
    *,
    floor: float = 1e-8,
) -> torch.Tensor:
    """In-sample seasonal-naive MAE per example: mean |y_t - y_{t-m}| over the
    context window, the standard MASE denominator.

    `context`/`valid`: [B, L]. `periods`: [B] integer seasonal lag m per
    example. A pair contributes only where both endpoints are valid.

    When m does not fit in the context window the lag falls back to 1, the
    plain random-walk naive baseline. Some corpus frequencies imply a seasonal
    period far longer than the context (e.g. "4S" implies a daily cycle of
    21,600 steps against a 512-step context); dropping those windows would
    remove entire datasets from the dispersion metrics, which distorts Gini
    more than using a shorter lag does. Every window of a dataset shares that
    dataset's frequency, so the lag is constant within a source either way.

    Examples that are exactly constant yield a denominator below `floor` and
    are returned as NaN so callers drop them, rather than emitting a
    divide-by-near-zero MASE that would dominate any mean or Gini.
    """
    length = context.shape[1]
    idx = torch.arange(length, device=context.device).unsqueeze(0)
    lag = periods.clamp_min(1)
    lag = torch.where(lag >= length, torch.ones_like(lag), lag).unsqueeze(1)
    src = idx - lag
    gather_idx = src.clamp_min(0)
    lagged = torch.gather(context, 1, gather_idx)
    lagged_valid = torch.gather(valid, 1, gather_idx)
    pair_valid = valid * lagged_valid * (src >= 0).to(valid.dtype)
    ae = (context - lagged).abs() * pair_valid
    denom = pair_valid.sum(dim=1)
    naive = ae.sum(dim=1) / denom.clamp_min(1.0)
    unusable = (denom < 1.0) | (naive < floor)
    return naive.masked_fill(unusable, float("nan"))


def pinball_loss(
    pred_quantiles: torch.Tensor,
    target: torch.Tensor,
    quantiles: list[float],
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Pinball (quantile) loss.

    pred_quantiles: [..., Q] matching `quantiles`. target: [...] (broadcasts
    against the Q dimension). Returns per-example mean pinball loss over Q and
    the trailing dims when `reduction='none'` (shape `[batch]`).
    """
    q = torch.as_tensor(
        quantiles, dtype=pred_quantiles.dtype, device=pred_quantiles.device
    )
    diff = target.unsqueeze(-1) - pred_quantiles
    loss = torch.maximum(q * diff, (q - 1.0) * diff)
    per_example = loss.mean(dim=tuple(range(1, loss.ndim)))
    if reduction == "none":
        return per_example
    if reduction == "mean":
        return per_example.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a 1-D array of non-negative per-source values.

    Uses the sorted-index formula G = (2 * sum(i * x_i)) / (n * sum(x)) - (n+1)/n
    for x sorted ascending, i = 1..n. Returns 0.0 for a single source (no
    dispersion is measurable with n=1) and requires non-negative inputs since
    the standard Gini formula assumes a non-negative quantity (error/loss here).
    """
    x = np.asarray(values, dtype=np.float64)
    if np.any(x < 0):
        raise ValueError(
            "gini_coefficient requires non-negative values (errors/losses)"
        )
    n = x.shape[0]
    if n <= 1:
        return 0.0
    if x.sum() == 0:
        return 0.0
    x_sorted = np.sort(x)
    index = np.arange(1, n + 1, dtype=np.float64)
    return float(
        (2.0 * np.sum(index * x_sorted)) / (n * x_sorted.sum()) - (n + 1.0) / n
    )


def dispersion_metrics(per_source_error: dict[str, float]) -> dict:
    """Gini, unweighted mean, and source count for one breakdown (dataset/
    domain/frequency) at one checkpoint. Does NOT compute the pooled
    natural-mixture-weighted global error -- that must be computed directly
    from per-example losses (see `pooled_mean`), not derived from per-source
    means, since the two are only equal for a balanced validation set.
    """
    values = np.array(list(per_source_error.values()), dtype=np.float64)
    values = values[np.isfinite(values)]  # MASE drops sources with no usable windows
    return {
        "gini": gini_coefficient(values),
        "unweighted_mean": float(values.mean()) if values.size else float("nan"),
        "n_sources": int(values.size),
    }


def pooled_mean(per_example_error: np.ndarray) -> float:
    """Natural-mixture-weighted global error: the plain mean over every example
    in its natural (imbalanced) proportion, as opposed to `unweighted_mean` in
    `dispersion_metrics`, which averages per-source means and therefore weights
    every source equally regardless of size.

    Ignores non-finite entries so a MASE array with dropped windows pools over
    the windows that do have a usable seasonal-naive denominator."""
    return float(np.nanmean(per_example_error))


def group_mean_by_source(
    per_example_error: np.ndarray, source_ids: np.ndarray
) -> dict[str, float]:
    """Per-source mean error, for feeding into `dispersion_metrics`.

    Non-finite per-example entries (MASE windows without a usable
    seasonal-naive denominator) are ignored; a source with no usable window at
    all yields NaN and is dropped by `dispersion_metrics`.
    """
    out: dict[str, float] = {}
    source_ids = np.asarray(source_ids)
    for source in np.unique(source_ids):
        values = per_example_error[source_ids == source]
        usable = values[np.isfinite(values)]
        out[str(source)] = float(usable.mean()) if usable.size else float("nan")
    return out


def group_median_by_source(
    per_example_error: np.ndarray, source_ids: np.ndarray
) -> dict[str, float]:
    """Per-source *median* error, the outlier-robust counterpart to
    `group_mean_by_source`.

    Needed because per-window normalized error is heavy-tailed on real data:
    a sparse intermittent series (e.g. retail unit sales that are mostly zero)
    can have a context standard deviation far smaller than a rare spike, so
    dividing by it sends one window's nMSE to ~1e7 and that single window then
    determines its dataset's mean and the corpus Gini. The median answers the
    same question ("how well is this source fit") without letting one window
    set the answer.
    """
    out: dict[str, float] = {}
    source_ids = np.asarray(source_ids)
    for source in np.unique(source_ids):
        values = per_example_error[source_ids == source]
        usable = values[np.isfinite(values)]
        out[str(source)] = float(np.median(usable)) if usable.size else float("nan")
    return out


def log_mse_auc(
    steps: np.ndarray, mse_values: np.ndarray, cutoff_step: int = TROUGH_STEP_CUTOFF
) -> float:
    """Trapezoidal area under log10(MSE) vs step, restricted to steps <= cutoff.

    Matches the synthetic loss-space convention of reporting AUC through the
    first 2,000 steps (see notes/00-experiments-log.md and the plan's Gini
    table instruction to report "through the first 2,000 steps").
    """
    steps = np.asarray(steps, dtype=np.float64)
    mse_values = np.asarray(mse_values, dtype=np.float64)
    keep = steps <= cutoff_step
    if keep.sum() < 2:
        raise ValueError(f"need >=2 points with step <= {cutoff_step} to integrate AUC")
    s = steps[keep]
    order = np.argsort(s)
    s = s[order]
    log_mse = np.log10(np.clip(mse_values[keep][order], 1e-12, None))
    return float(np.trapezoid(log_mse, s))


def steps_to_threshold(
    steps: np.ndarray, mse_values: np.ndarray, threshold: float
) -> int | None:
    """First step at which mse_values <= threshold, or None if never reached."""
    steps = np.asarray(steps)
    mse_values = np.asarray(mse_values)
    order = np.argsort(steps)
    hit = np.nonzero(mse_values[order] <= threshold)[0]
    if hit.size == 0:
        return None
    return int(steps[order][hit[0]])


def grad_norm(parameters) -> float:
    """L2 norm of gradients across `parameters` (call before/after clipping)."""
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm(2) ** 2)
    return total**0.5


def backward_with_safe_gradient_clipping(
    loss: torch.Tensor,
    parameters,
    max_norm: float,
    tracked_parameters=(),
) -> dict[str, float | bool]:
    """Backpropagate and clip without overflowing the gradient-norm reduction.

    A temporary scalar multiplier keeps the backward pass representable. The
    multiplier is removed analytically when computing the original gradient
    norm and final clipping coefficient, so the optimizer receives the same
    clipped gradient direction and magnitude as an exact unscaled reduction.
    """
    parameters = list(parameters)
    tracked_parameters = list(tracked_parameters)
    loss_value = float(loss.detach().abs())
    if not np.isfinite(loss_value):
        raise RuntimeError("training loss is not finite")

    backward_scale = 1.0 / max(1.0, loss_value)
    (loss * backward_scale).backward()
    scaled_norm = grad_norm(parameters)
    scaled_tracked_norm = grad_norm(tracked_parameters)
    if not np.isfinite(scaled_norm) or not np.isfinite(scaled_tracked_norm):
        raise RuntimeError("scaled gradient norm is not finite")

    norm_before_clip = scaled_norm / backward_scale
    tracked_norm_before_clip = scaled_tracked_norm / backward_scale
    clip_coefficient = min(1.0, max_norm / (norm_before_clip + 1e-6))
    gradient_multiplier = clip_coefficient / backward_scale
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(gradient_multiplier)

    return {
        "tracked_norm_before_clip": tracked_norm_before_clip,
        "total_norm_before_clip": norm_before_clip,
        "total_norm_after_clip": grad_norm(parameters),
        "clipped": clip_coefficient < 1.0,
    }

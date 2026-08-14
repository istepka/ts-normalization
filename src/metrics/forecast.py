"""Per-window forecast error definitions, scale-free by construction."""

import torch


def per_sample_nmse(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """Normalized-space MSE per sample; the common metric for both runs."""
    return (z_pred - z_target).pow(2).mean(dim=1)


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

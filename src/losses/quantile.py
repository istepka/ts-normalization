"""Quantile objectives.

Two conventions are in use and they differ by more than a constant, so the
adapters that need each one keep calling their own. `crps_quantile_loss` is
twice `pinball_loss` and sums over quantile levels rather than averaging.
"""

import torch


def pinball_loss(
    pred_quantiles: torch.Tensor,
    target: torch.Tensor,
    quantiles: list[float],
    *,
    valid: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Pinball (quantile) loss, averaged over quantiles.

    `pred_quantiles`: [..., Q] matching `quantiles`. `target`: [...]
    (broadcasts against the Q dimension). `valid`, when given, has `target`'s
    shape and restricts the average to valid positions (1 = included);
    without it every position counts. `reduction='none'` returns the
    per-example loss (shape `[batch]`).
    """
    q = torch.as_tensor(
        quantiles, dtype=pred_quantiles.dtype, device=pred_quantiles.device
    )
    diff = target.unsqueeze(-1) - pred_quantiles
    loss = torch.maximum(q * diff, (q - 1.0) * diff)
    if valid is None:
        per_example = loss.mean(dim=tuple(range(1, loss.ndim)))
    else:
        loss = loss * valid.unsqueeze(-1)
        denom = valid.sum(dim=tuple(range(1, valid.ndim))).clamp_min(1.0)
        per_example = loss.sum(dim=tuple(range(1, loss.ndim))) / (
            denom * len(quantiles)
        )
    if reduction == "none":
        return per_example
    if reduction == "mean":
        return per_example.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def crps_quantile_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    """Chronos-2 / Moirai-2.0 quantile loss, summed over quantiles.

    `predictions`: [B, Q, H] with the quantile axis in the middle, unlike
    `pinball_loss`. `target`/`valid`: [B, H]. Returns a per-example loss of
    shape `[B]`.
    """
    target = target.unsqueeze(1)
    valid = valid.unsqueeze(1)
    quantiles = quantiles.view(1, -1, 1)
    loss = 2.0 * torch.abs(
        (target - predictions)
        * ((target <= predictions).to(predictions.dtype) - quantiles)
    )
    return (loss * valid).mean(dim=-1).sum(dim=-1)

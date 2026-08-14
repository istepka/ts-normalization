"""Masked pointwise objectives shared by every model adapter.

Both take a mask with the same rank as the prediction (1 = included) and
reduce to a per-example mean over the masked positions, so per-source
dispersion metrics see one number per window regardless of model.
"""

import torch


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

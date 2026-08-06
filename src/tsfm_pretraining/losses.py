"""Loss-space primitives and dispersion/equity metrics shared by both TSFM stages.

Both the MOMENT and TimesFM adapters call into this module rather than each
re-deriving MSE, pinball loss, or the Gini/exposure bookkeeping, so the two
stages report metrics on exactly the same definitions (see
notes/05-timesfm-pretraining-loss-space-plan.md, "Dispersion and equity
metrics").
"""

import numpy as np
import torch

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
    return {
        "gini": gini_coefficient(values),
        "unweighted_mean": float(values.mean()) if values.size else float("nan"),
        "n_sources": int(values.size),
    }


def pooled_mean(per_example_error: np.ndarray) -> float:
    """Natural-mixture-weighted global error: the plain mean over every example
    in its natural (imbalanced) proportion, as opposed to `unweighted_mean` in
    `dispersion_metrics`, which averages per-source means and therefore weights
    every source equally regardless of size."""
    return float(np.mean(per_example_error))


def group_mean_by_source(
    per_example_error: np.ndarray, source_ids: np.ndarray
) -> dict[str, float]:
    """Per-source mean error, for feeding into `dispersion_metrics`."""
    out: dict[str, float] = {}
    source_ids = np.asarray(source_ids)
    for source in np.unique(source_ids):
        out[str(source)] = float(per_example_error[source_ids == source].mean())
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

"""How fast a run's loss curve comes down, summarized as single numbers."""

import numpy as np

TROUGH_STEP_CUTOFF = 2000  # matches the synthetic loss-space Gini table convention


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

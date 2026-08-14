"""Dispersion and equity metrics over per-source errors.

Every stage reports these on the same definitions, so a Gini from the
synthetic toy and a Gini from a TSFM run mean the same thing.
"""

import numpy as np


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

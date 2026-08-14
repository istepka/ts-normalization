"""Base plotting machinery shared by every figure and table script."""

from src.plotting.core.figures import (
    GLOBAL_STYLE,
    PAPER_RCPARAMS,
    apply_paper_style,
    bottom_legend,
    mean_ci,
    save_figure,
    style_step_axis,
)
from src.plotting.core.palette import (
    ACCENT,
    CATEGORICAL,
    CATEGORICAL_ORDER,
    PRIMARY,
    SECONDARY,
    TEAL,
    TEXT,
    apply_palette,
    categorical_colors,
)
from src.plotting.core.tsfm_runs import (
    COLORS,
    LABELS,
    LOSS_SPACES,
    MODELS,
    capped_gini,
    final_dataset_mase,
    load_histories,
    metric_values,
    per_dataset_curves,
    seed_metric_curves,
)

__all__ = [
    "ACCENT",
    "CATEGORICAL",
    "CATEGORICAL_ORDER",
    "COLORS",
    "GLOBAL_STYLE",
    "LABELS",
    "LOSS_SPACES",
    "MODELS",
    "PAPER_RCPARAMS",
    "PRIMARY",
    "SECONDARY",
    "TEAL",
    "TEXT",
    "apply_palette",
    "apply_paper_style",
    "bottom_legend",
    "capped_gini",
    "categorical_colors",
    "final_dataset_mase",
    "load_histories",
    "mean_ci",
    "metric_values",
    "per_dataset_curves",
    "save_figure",
    "seed_metric_curves",
    "style_step_axis",
]

apply_palette()
apply_paper_style()

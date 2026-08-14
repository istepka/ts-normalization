"""Generate title-free model panels and a shared legend for Figure 6."""

import matplotlib.pyplot as plt

from src.plotting.core import (
    COLORS,
    LABELS,
    LOSS_SPACES,
    MODELS,
    mean_ci,
    save_figure,
    seed_metric_curves,
    style_step_axis,
)
from src.plotting.scripts.plot_tsfm_natural_convergence import OUTPUT_DIR


def main() -> None:
    series = {}
    lower_bounds = []
    upper_bounds = []
    for model_name, model in MODELS.items():
        series[model_name] = {}
        for loss_space in LOSS_SPACES:
            steps, values = seed_metric_curves(
                model["histories"][loss_space], "dataset_median"
            )
            mean, lower, upper = mean_ci(values)
            series[model_name][loss_space] = (steps, mean, lower, upper)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

    y_min = min(float(values.min()) for values in lower_bounds)
    y_max = max(float(values.max()) for values in upper_bounds)
    padding = 0.04 * (y_max - y_min)
    for index, (model_name, model_series) in enumerate(series.items()):
        fig, axis = plt.subplots(figsize=(1.75, 1.5))
        for loss_space in LOSS_SPACES:
            steps, mean, lower, upper = model_series[loss_space]
            axis.plot(
                steps,
                mean,
                color=COLORS[loss_space],
                linewidth=1.35,
            )
            axis.fill_between(
                steps,
                lower,
                upper,
                color=COLORS[loss_space],
                alpha=0.14,
                linewidth=0,
            )
        axis.set_ylim(y_min - padding, y_max + padding)
        axis.set_xlabel("Training step", fontsize=7)
        if index == 0:
            axis.set_ylabel("Median per-dataset MASE", fontsize=7)
        else:
            axis.tick_params(labelleft=False)
        style_step_axis(axis, end_step=int(steps[-1]))
        fig.subplots_adjust(left=0.22, right=0.98, bottom=0.25, top=0.97)
        stem = f"paper_full_median_mase_{MODELS[model_name]['key']}"
        save_figure(fig, OUTPUT_DIR / stem, bbox_inches=None)

    legend_fig = plt.figure(figsize=(3.1, 0.25))
    handles = [
        plt.Line2D([], [], color=COLORS[loss_space], linewidth=1.35)
        for loss_space in LOSS_SPACES
    ]
    legend_fig.legend(
        handles,
        [LABELS[loss_space] for loss_space in LOSS_SPACES],
        loc="center",
        frameon=False,
        ncol=2,
        fontsize=7,
    )
    save_figure(legend_fig, OUTPUT_DIR / "paper_full_median_mase_legend")


if __name__ == "__main__":
    main()

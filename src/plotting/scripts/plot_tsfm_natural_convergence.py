"""Plot paper-ready natural-scale TSFM convergence figures."""

from pathlib import Path

import matplotlib.pyplot as plt

from src.plotting.core import (
    COLORS,
    LABELS,
    LOSS_SPACES,
    MODELS,
    bottom_legend,
    mean_ci,
    save_figure,
    seed_metric_curves,
    style_step_axis,
)

OUTPUT_DIR = Path(
    "outputs/2026-08-13/visualizations/tsfm_pretraining/four_model_natural_convergence"
)


def plot_full(metric: str, ylabel: str, stem: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15), sharey=True)
    for axis, (title, model) in zip(axes, MODELS.items(), strict=True):
        for loss_space in LOSS_SPACES:
            steps, values = seed_metric_curves(model["histories"][loss_space], metric)
            mean, lower, upper = mean_ci(values)
            axis.plot(
                steps,
                mean,
                color=COLORS[loss_space],
                label=LABELS[loss_space],
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
        axis.set_title(title, fontsize=8)
        axis.set_xlabel("Training step", fontsize=7)
        style_step_axis(axis, end_step=int(steps[-1]))
    axes[0].set_ylabel(ylabel, fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    bottom_legend(fig, handles, labels, ncol=2)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.31, top=0.90, wspace=0.12)
    save_figure(fig, OUTPUT_DIR / stem)


def main() -> None:
    plot_full(
        "dataset_median",
        "Median per-dataset MASE",
        "paper_full_median_mase",
    )
    plot_full(
        "global_mean",
        "Mean MASE across windows",
        "paper_full_mean_mase",
    )


if __name__ == "__main__":
    main()

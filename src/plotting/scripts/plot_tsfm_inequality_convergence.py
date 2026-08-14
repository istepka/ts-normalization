"""Plot capped Gini and per-dataset MASE trajectories over TSFM training."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.core import (
    COLORS,
    LABELS,
    LOSS_SPACES,
    MODELS,
    bottom_legend,
    capped_gini,
    load_histories,
    mean_ci,
    per_dataset_curves,
    save_figure,
    style_step_axis,
)

OUTPUT_DIR = Path(
    "outputs/2026-08-13/visualizations/tsfm_pretraining/"
    "inequality_and_dataset_convergence"
)


def plot_capped_gini() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15), sharey=True)
    for axis, (model_name, model) in zip(axes, MODELS.items(), strict=True):
        for loss_space in LOSS_SPACES:
            histories = load_histories(model["histories"][loss_space], end_step=30_000)
            steps, _, values = per_dataset_curves(histories)
            seed_gini = np.asarray(
                [[capped_gini(checkpoint) for checkpoint in seed] for seed in values]
            )
            mean, lower, upper = mean_ci(seed_gini)
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
        axis.set_title(model_name, fontsize=8)
        axis.set_xlabel("Training step", fontsize=7)
        style_step_axis(axis)
    axes[0].set_ylabel("Capped MASE Gini", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    bottom_legend(fig, handles, labels, ncol=2)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.31, top=0.90, wspace=0.12)
    save_figure(fig, OUTPUT_DIR / "capped_gini_convergence")


def plot_per_dataset(yscale: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.1), sharex=True, sharey=True)
    for column, (model_name, model) in enumerate(MODELS.items()):
        for row, loss_space in enumerate(LOSS_SPACES):
            axis = axes[row, column]
            histories = load_histories(model["histories"][loss_space], end_step=30_000)
            steps, datasets, values = per_dataset_curves(histories)
            mean_per_dataset = values.mean(axis=0)
            worst_indices = np.argsort(mean_per_dataset[-1])[-2:]
            keep = np.ones(len(datasets), dtype=bool)
            keep[worst_indices] = False
            excluded = [datasets[index] for index in worst_indices]
            print(f"{model_name} {loss_space} excluded: {excluded}")
            displayed = mean_per_dataset[:, keep]
            for dataset_values in displayed.T:
                axis.plot(
                    steps,
                    dataset_values,
                    color=COLORS[loss_space],
                    alpha=0.20,
                    linewidth=0.65,
                )
            axis.plot(
                steps,
                np.median(displayed, axis=1),
                color="black",
                linewidth=1.25,
                label="Dataset median",
            )
            axis.set_yscale(yscale)
            if row == 0:
                axis.set_title(model_name, fontsize=8)
            if column == 0:
                axis.set_ylabel(
                    f"{LABELS[loss_space]}\nPer-dataset MASE",
                    fontsize=7,
                )
            if row == 1:
                axis.set_xlabel("Training step", fontsize=7)
            style_step_axis(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    bottom_legend(fig, handles, labels, ncol=1)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.17, top=0.94, wspace=0.12)
    save_figure(fig, OUTPUT_DIR / f"per_dataset_mase_convergence_{yscale}")


def plot_cross_dataset_variance(exclude_worst: bool) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15), sharey=True)
    for axis, (model_name, model) in zip(axes, MODELS.items(), strict=True):
        for loss_space in LOSS_SPACES:
            histories = load_histories(model["histories"][loss_space], end_step=30_000)
            steps, datasets, values = per_dataset_curves(histories)
            if exclude_worst:
                mean_per_dataset = values.mean(axis=0)
                worst_indices = np.argsort(mean_per_dataset[-1])[-2:]
                keep = np.ones(len(datasets), dtype=bool)
                keep[worst_indices] = False
                values = values[:, :, keep]
            mean, lower, upper = mean_ci(np.var(values, axis=2))
            axis.plot(
                steps,
                mean,
                color=COLORS[loss_space],
                label=LABELS[loss_space],
                linewidth=1.35,
            )
            axis.fill_between(
                steps,
                np.maximum(lower, 0),
                upper,
                color=COLORS[loss_space],
                alpha=0.14,
                linewidth=0,
            )
        axis.set_title(model_name, fontsize=8)
        axis.set_xlabel("Training step", fontsize=7)
        style_step_axis(axis)
    axes[0].set_ylabel("Variance across dataset MASE", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    bottom_legend(fig, handles, labels, ncol=2)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.31, top=0.90, wspace=0.12)
    suffix = "excluding_worst_two" if exclude_worst else "all_datasets"
    save_figure(fig, OUTPUT_DIR / f"cross_dataset_mase_variance_{suffix}")


def main() -> None:
    plot_capped_gini()
    plot_per_dataset("linear")
    plot_per_dataset("log")
    plot_cross_dataset_variance(exclude_worst=False)
    plot_cross_dataset_variance(exclude_worst=True)


if __name__ == "__main__":
    main()

"""Plot capped Gini and per-dataset MASE trajectories for the 80k in-domain run.

Same figures as `plot_tsfm_inequality_convergence.py`, but over the
variable-geometry in-domain continuation (jobs 32875 / 32878, resumed from
32797 / 32807 to 80,000 updates) and restricted to Chronos-2 and Moirai-2.0,
the only two architectures in that run.

Every figure is rendered once over all datasets and once with the two
highest-error datasets (per model and loss space, by final-step mean) removed,
into sibling `all_datasets/` and `excluding_worst_two/` folders. The
cross-dataset variance plot additionally gets an `excluding_worst_four/`
variant.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.core import (
    COLORS,
    LOSS_SPACES,
    bottom_legend,
    capped_gini,
    load_histories,
    mean_ci,
    per_dataset_curves,
    save_figure,
    seed_metric_curves,
    style_step_axis,
)

END_STEP = 80_000
TICK_STEP = 20_000
LABELS = {
    "normalized": "Scale-invariant loss",
    "original": "Scale-contaminated loss",
}

CHRONOS_ROOT = Path(
    "outputs/2026-08-18/experiments/tsfm_pretraining/chronos2/gifteval_chronos2_32875"
)
MOIRAI_ROOT = Path(
    "outputs/2026-08-18/experiments/tsfm_pretraining/moirai2/gifteval_moirai2_32878"
)

MODELS = {
    "Chronos-2": {
        "histories": {
            condition: [
                CHRONOS_ROOT
                / f"seed{seed}_chronos2_{condition}_natural"
                / "history.json"
                for seed in range(4)
            ]
            for condition in LOSS_SPACES
        },
    },
    "Moirai-2.0": {
        "histories": {
            condition: [
                MOIRAI_ROOT / f"seed{seed}_moirai2_{condition}_natural" / "history.json"
                for seed in range(4)
            ]
            for condition in LOSS_SPACES
        },
    },
}

BASE_OUTPUT_DIR = Path(
    "outputs/2026-08-18/visualizations/tsfm_pretraining/"
    "inequality_and_dataset_convergence_80k_indomain"
)


FOLDER_NAMES = {0: "all_datasets", 2: "excluding_worst_two", 4: "excluding_worst_four"}


def output_dir(exclude_n: int) -> Path:
    return BASE_OUTPUT_DIR / FOLDER_NAMES[exclude_n]


def worst_n_mask(values: np.ndarray, datasets: list[str], n: int) -> np.ndarray:
    """Boolean keep-mask dropping the `n` datasets with the highest final-step
    mean, computed over seeds."""
    mean_per_dataset = values.mean(axis=0)
    worst_indices = np.argsort(mean_per_dataset[-1])[-n:]
    keep = np.ones(len(datasets), dtype=bool)
    keep[worst_indices] = False
    return keep


def plot_capped_gini(exclude_n: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.8, 2.15), sharey=True)
    for axis, (model_name, model) in zip(axes, MODELS.items(), strict=True):
        for loss_space in LOSS_SPACES:
            histories = load_histories(
                model["histories"][loss_space], end_step=END_STEP
            )
            steps, datasets, values = per_dataset_curves(histories)
            if exclude_n:
                values = values[:, :, worst_n_mask(values, datasets, exclude_n)]
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
        style_step_axis(axis, end_step=END_STEP, tick_step=TICK_STEP)
    axes[0].set_ylabel("Capped MASE Gini", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    bottom_legend(fig, handles, labels, ncol=2)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.31, top=0.90, wspace=0.12)
    save_figure(fig, output_dir(exclude_n) / "capped_gini_convergence")


def plot_per_dataset(yscale: str, exclude_n: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(3.8, 4.1), sharex=True, sharey=True)
    for column, (model_name, model) in enumerate(MODELS.items()):
        for row, loss_space in enumerate(LOSS_SPACES):
            axis = axes[row, column]
            histories = load_histories(
                model["histories"][loss_space], end_step=END_STEP
            )
            steps, datasets, values = per_dataset_curves(histories)
            mean_per_dataset = values.mean(axis=0)
            if exclude_n:
                keep = worst_n_mask(values, datasets, exclude_n)
                excluded = [
                    dataset for dataset, kept in zip(datasets, keep) if not kept
                ]
                print(f"{model_name} {loss_space} excluded: {excluded}")
            else:
                keep = np.ones(len(datasets), dtype=bool)
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
            style_step_axis(axis, end_step=END_STEP, tick_step=TICK_STEP)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    bottom_legend(fig, handles, labels, ncol=1)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.17, top=0.94, wspace=0.12)
    save_figure(fig, output_dir(exclude_n) / f"per_dataset_mase_convergence_{yscale}")


def plot_per_dataset_overlay(yscale: str, exclude_n: int) -> None:
    """The per-dataset plot with both loss spaces overlaid on one row."""
    fig, axes = plt.subplots(1, 2, figsize=(3.8, 2.3), sharex=True, sharey=True)
    for axis, (model_name, model) in zip(axes, MODELS.items(), strict=True):
        for loss_space in LOSS_SPACES:
            histories = load_histories(
                model["histories"][loss_space], end_step=END_STEP
            )
            steps, datasets, values = per_dataset_curves(histories)
            mean_per_dataset = values.mean(axis=0)
            if exclude_n:
                keep = worst_n_mask(values, datasets, exclude_n)
            else:
                keep = np.ones(len(datasets), dtype=bool)
            displayed = mean_per_dataset[:, keep]
            for dataset_values in displayed.T:
                axis.plot(
                    steps,
                    dataset_values,
                    color=COLORS[loss_space],
                    alpha=0.15,
                    linewidth=0.6,
                )
            axis.plot(
                steps,
                np.median(displayed, axis=1),
                color=COLORS[loss_space],
                linewidth=1.35,
                label=LABELS[loss_space],
            )
        axis.set_yscale(yscale)
        axis.set_title(model_name, fontsize=8)
        axis.set_xlabel("Training step", fontsize=7)
        style_step_axis(axis, end_step=END_STEP, tick_step=TICK_STEP)
    axes[0].set_ylabel("Per-dataset MASE\n(median across datasets)", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    bottom_legend(fig, handles, labels, ncol=2)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.30, top=0.90, wspace=0.12)
    save_figure(
        fig,
        output_dir(exclude_n) / f"per_dataset_mase_convergence_{yscale}_overlay",
    )


def plot_cross_dataset_variance(exclude_n: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.8, 2.15), sharey=True)
    for axis, (model_name, model) in zip(axes, MODELS.items(), strict=True):
        for loss_space in LOSS_SPACES:
            histories = load_histories(
                model["histories"][loss_space], end_step=END_STEP
            )
            steps, datasets, values = per_dataset_curves(histories)
            if exclude_n:
                values = values[:, :, worst_n_mask(values, datasets, exclude_n)]
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
        style_step_axis(axis, end_step=END_STEP, tick_step=TICK_STEP)
    axes[0].set_ylabel("Variance across dataset MASE", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    bottom_legend(fig, handles, labels, ncol=2)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.31, top=0.90, wspace=0.12)
    save_figure(fig, output_dir(exclude_n) / "cross_dataset_mase_variance")


def plot_pooled_metric(metric: str, ylabel: str, stem: str) -> None:
    """A single pooled validation curve per model, not broken out by dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(3.8, 2.15), sharey=True)
    for axis, (model_name, model) in zip(axes, MODELS.items(), strict=True):
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
        axis.set_title(model_name, fontsize=8)
        axis.set_xlabel("Training step", fontsize=7)
        style_step_axis(axis, end_step=END_STEP, tick_step=TICK_STEP)
    axes[0].set_ylabel(ylabel, fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    bottom_legend(fig, handles, labels, ncol=2)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.31, top=0.90, wspace=0.12)
    save_figure(fig, BASE_OUTPUT_DIR / "pooled_metrics" / stem)


def main() -> None:
    for exclude_n in (0, 2):
        plot_capped_gini(exclude_n)
        plot_per_dataset("linear", exclude_n)
        plot_per_dataset("log", exclude_n)
        plot_per_dataset_overlay("linear", exclude_n)
    for exclude_n in (0, 2, 4):
        plot_cross_dataset_variance(exclude_n)
    plot_pooled_metric("global_mean", "Pooled MASE", "pooled_mase_convergence")
    plot_pooled_metric(
        "dataset_median", "Median per-dataset MASE", "median_mase_convergence"
    )


if __name__ == "__main__":
    main()

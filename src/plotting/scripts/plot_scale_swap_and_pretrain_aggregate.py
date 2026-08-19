"""Two companion figures for the paper, meant as a pair of subfigures.

The first collapses `scale_swap_permutation_by_dataset` across its eight
datasets: for each of the 15 complementary assignment pairs we take the
median nMSE across datasets, then plot the mean and 95% Student-t interval of
that median across pairs, one line per (loss space, scale) combination. The
second is the 80k in-domain median-per-dataset MASE convergence plot.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.core import (
    COLORS,
    LOSS_SPACES,
    bottom_legend,
    mean_ci,
    style_step_axis,
)
from src.plotting.core import save_figure as save_base_figure
from src.plotting.scripts.plot_tsfm_80k_indomain_convergence import (
    MODELS as PRETRAIN_MODELS,
)
from src.plotting.scripts.plot_tsfm_80k_indomain_convergence import (
    seed_metric_curves,
)
from src.plotting.scripts.reproducibility.real_scale_swap import (
    aggregate_permutation_campaign as scale_swap,
)

SCALE_SWAP_CAMPAIGN_ROOT = Path(
    "outputs/2026-07-29/experiments/legacy_runs/scale_swap_permutations"
)
SCALE_SWAP_NORMALIZED_ROOT = Path(
    "outputs/2026-08-05/experiments/legacy_runs/scale_swap_permutations_normalized"
)

OUTPUT_DIR = Path(
    "outputs/2026-08-18/visualizations/tsfm_pretraining/"
    "scale_swap_and_pretrain_aggregate"
)

# The two figures sit side by side in the paper as a 40%/58% split of the
# 6.3in text width (article class, 1.1in margins on 8.5in letter paper), with
# the remaining 2% going to the \hfill gap between them. Sizing each canvas
# to its actual target width, rather than generating both at one width and
# letting LaTeX rescale unevenly, keeps font and marker sizes consistent
# between the two subfigures instead of (a) shrinking more than (b).
TEXT_WIDTH_IN = 6.3
SCALE_SWAP_WIDTH_IN = 0.40 * TEXT_WIDTH_IN
PRETRAIN_WIDTH_IN = 0.58 * TEXT_WIDTH_IN
FIGURE_HEIGHT_IN = 2.3

PRETRAIN_LABELS = {
    "normalized": "Scale-invariant loss",
    "original": "Scale-contaminated loss",
}


def plot_scale_swap_aggregate() -> None:
    campaigns = {
        scale_swap.NORMALIZED_MODE: scale_swap.load_campaign(
            SCALE_SWAP_NORMALIZED_ROOT,
            scale_swap.NORMALIZED_MODE,
            scale_swap.PAIR_ONE_NORMALIZED_PATHS,
        ),
        scale_swap.MODE: scale_swap.load_campaign(
            SCALE_SWAP_CAMPAIGN_ROOT,
            scale_swap.MODE,
            scale_swap.PAIR_ONE_ORIGINAL_PATHS,
        ),
    }
    steps = campaigns[scale_swap.MODE]["steps"]
    modes = (
        (scale_swap.NORMALIZED_MODE, "normalized", "Scale-invariant loss"),
        (scale_swap.MODE, "original", "Scale-contaminated loss"),
    )
    # Color keys the loss space, matching the blue/red used for it in the
    # companion subfigure. Linestyle and marker key the scale probe together,
    # since the scale-invariant pair overlaps almost exactly and a plain
    # dotted line disappears into the solid one wherever that happens.
    scale_styles = (
        ("low", "$b=1$", "-", "o"),
        ("high", "$b=10$", "--", "^"),
    )

    # The pre-2k region is densely sampled to resolve early transients, so
    # markers there would clump into a solid smear. Marker indices are drawn
    # only from the sparser post-2k tail.
    marked_indices = np.flatnonzero(steps >= 2_000)

    fig, axis = plt.subplots(figsize=(SCALE_SWAP_WIDTH_IN, FIGURE_HEIGHT_IN))
    handles_by_mode = {}
    for mode, color_key, mode_label in modes:
        campaign = campaigns[mode]
        color = COLORS[color_key]
        handles_by_mode[mode] = []
        for key, scale_label, linestyle, marker in scale_styles:
            median_per_pair = np.median(campaign[key], axis=1)
            mean, lower, upper = mean_ci(median_per_pair)
            offset = 2 if key == "high" else 0
            (line,) = axis.plot(
                steps,
                mean,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=list(marked_indices[offset::5]),
                markersize=6,
                markeredgewidth=0,
                linewidth=1.35,
                alpha=1.0,
                label=scale_label,
            )
            handles_by_mode[mode].append(line)
            axis.fill_between(steps, lower, upper, color=color, alpha=0.14, linewidth=0)
    axis.set_ylabel("Median per-dataset nMSE", fontsize=7)
    axis.set_xlabel("Training step", fontsize=7)
    style_step_axis(axis, end_step=30_000, tick_step=10_000)

    invariant_legend = fig.legend(
        handles=handles_by_mode[scale_swap.NORMALIZED_MODE],
        title="Scale-invariant loss",
        loc="lower left",
        bbox_to_anchor=(0.02, 0.0),
        frameon=False,
        fontsize=6.5,
        title_fontsize=6.5,
        labelspacing=0.25,
    )
    fig.add_artist(invariant_legend)
    fig.legend(
        handles=handles_by_mode[scale_swap.MODE],
        title="Scale-contaminated loss",
        loc="lower right",
        bbox_to_anchor=(0.98, 0.0),
        frameon=False,
        fontsize=6.5,
        title_fontsize=6.5,
        labelspacing=0.25,
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.36, top=0.96)
    save_base_figure(fig, OUTPUT_DIR / "scale_swap_aggregate")


def plot_pretrain_median() -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(PRETRAIN_WIDTH_IN, FIGURE_HEIGHT_IN), sharey=False
    )
    for axis, (model_name, model) in zip(axes, PRETRAIN_MODELS.items(), strict=True):
        for loss_space in LOSS_SPACES:
            steps, values = seed_metric_curves(
                model["histories"][loss_space], "dataset_median"
            )
            mean, lower, upper = mean_ci(values)
            axis.plot(
                steps,
                mean,
                color=COLORS[loss_space],
                label=PRETRAIN_LABELS[loss_space],
                linewidth=1.35,
                alpha=0.95,
            )
            axis.fill_between(
                steps, lower, upper, color=COLORS[loss_space], alpha=0.14, linewidth=0
            )
        axis.set_title(model_name, fontsize=8)
        axis.set_xlabel("Training step", fontsize=7)
        style_step_axis(axis, end_step=80_000, tick_step=20_000)
    axes[0].set_ylabel("Median per-dataset MASE", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    bottom_legend(fig, handles, labels, ncol=2)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.28, top=0.89, wspace=0.32)
    save_base_figure(fig, OUTPUT_DIR / "pretrain_median_mase")


def main() -> None:
    plot_scale_swap_aggregate()
    plot_pretrain_median()


if __name__ == "__main__":
    main()

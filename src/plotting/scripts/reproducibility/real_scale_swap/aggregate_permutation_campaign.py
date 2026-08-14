"""Aggregate the 15-pair LR-adjusted scale-swap permutation campaign.

Paper: apd_loss_space_scale_swap.tex "Balanced assignment" (the p=0.042
paired t-test on dataset-level AUC) and Fig. loss-space-scale-swap-datasets.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import ttest_rel, wilcoxon

from src.plotting.core import PRIMARY, SECONDARY, mean_ci
from src.plotting.core import save_figure as save_base_figure
from src.scripts.permutation_schedule import (
    HIGH_SCALE,
    LOW_SCALE,
    assignment_pair,
    validate_schedule,
)

NORMALIZED_MODE = "normalized"
MODE = "original_lr_adjusted"
NUM_PAIRS = 15
NUM_DATASETS = 8
LEGACY_ROOT = Path("outputs/2026-07-17/experiments/legacy_runs")
PAIR_ONE_ORIGINAL_PATHS = (
    LEGACY_ROOT / "18985_scale_swap_lr_adjusted_a",
    LEGACY_ROOT / "18985_scale_swap_lr_adjusted_b",
)
PAIR_ONE_NORMALIZED_PATHS = (
    LEGACY_ROOT / "18948_scale_swap_a",
    LEGACY_ROOT / "18948_scale_swap_b",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign-root", type=Path, default=Path("outputs/scale_swap_permutations")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/scale_swap_permutation_analysis"),
    )
    parser.add_argument(
        "--normalized-campaign-root",
        type=Path,
        help="completed normalized-space control outputs for pairs 2 through 15",
    )
    return parser.parse_args()


def load_metrics(path: Path, mode: str) -> dict:
    with np.load(path / f"metrics_{mode}.npz") as data:
        names = [str(value) for value in data["names"]]
        steps = data["step"]
        nmse = data["nmse"]
    if len(names) != NUM_DATASETS:
        raise ValueError(f"expected {NUM_DATASETS} datasets in {path}")
    if not np.isfinite(nmse).all():
        raise ValueError(f"non-finite nMSE in {path}")
    return {"names": names, "steps": steps, "nmse": nmse}


def pair_paths(
    pair_index: int,
    campaign_root: Path,
    pair_one_paths: tuple[Path, Path],
) -> tuple[Path, Path]:
    if pair_index == 1:
        return pair_one_paths
    pair_tag = f"{pair_index:02d}"
    return (
        campaign_root / f"pair_{pair_tag}_a",
        campaign_root / f"pair_{pair_tag}_b",
    )


def load_campaign(
    campaign_root: Path,
    mode: str,
    pair_one_paths: tuple[Path, Path],
) -> dict:
    validate_schedule()
    low_curves = []
    high_curves = []
    reference_names = None
    reference_steps = None
    for pair_index in range(1, NUM_PAIRS + 1):
        paths = pair_paths(pair_index, campaign_root, pair_one_paths)
        variants = [load_metrics(path, mode) for path in paths]
        if reference_names is None:
            reference_names = variants[0]["names"]
            reference_steps = variants[0]["steps"]
        for path, variant in zip(paths, variants):
            if variant["names"] != reference_names or not np.array_equal(
                variant["steps"], reference_steps
            ):
                raise ValueError(f"incompatible metrics in {path}")

        assignment_a, assignment_b = assignment_pair(pair_index)
        assignments = (assignment_a, assignment_b)
        pair_low = np.empty((NUM_DATASETS, len(reference_steps)))
        pair_high = np.empty_like(pair_low)
        for dataset in range(NUM_DATASETS):
            for variant, assignment in zip(variants, assignments):
                curve = variant["nmse"][:, :, dataset].mean(axis=0)
                if assignment[dataset] == LOW_SCALE:
                    pair_low[dataset] = curve
                elif assignment[dataset] == HIGH_SCALE:
                    pair_high[dataset] = curve
                else:
                    raise ValueError("assignment contains an unexpected scale")
        low_curves.append(pair_low)
        high_curves.append(pair_high)
    return {
        "mode": mode,
        "names": reference_names,
        "steps": reference_steps,
        "low": np.stack(low_curves),
        "high": np.stack(high_curves),
    }


def linear_auc(curves: np.ndarray, steps: np.ndarray) -> np.ndarray:
    duration = steps[-1] - steps[0]
    return np.trapezoid(curves, steps, axis=-1) / duration


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def save_figure(fig, path: Path):
    save_base_figure(fig, path, dpi=150, tight_layout=True, bbox_inches=None)


def plot_aggregate(campaign: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    steps = campaign["steps"]
    for curves, label, color in (
        (campaign["low"], "assigned $b=1$", SECONDARY),
        (campaign["high"], "assigned $b=10$", PRIMARY),
    ):
        dataset_curves = curves.mean(axis=0)
        mean, lower, upper = mean_ci(dataset_curves)
        ax.plot(steps, mean, color=color, label=label)
        ax.fill_between(steps, lower, upper, color=color, alpha=0.2)
    ax.set_xlabel("step")
    ax.set_ylabel("nMSE")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    save_figure(fig, output_path)


def plot_by_dataset(campaigns: dict[str, dict], output_path: Path):
    modes = (
        (NORMALIZED_MODE, "Normalized-space loss", "--"),
        (MODE, "LR-adjusted original-space loss", "-"),
    )
    available_modes = [mode for mode, _, _ in modes if mode in campaigns]
    if not available_modes:
        raise ValueError("at least one campaign is required")
    reference = campaigns[available_modes[0]]
    for mode in available_modes[1:]:
        campaign = campaigns[mode]
        if campaign["names"] != reference["names"] or not np.array_equal(
            campaign["steps"], reference["steps"]
        ):
            raise ValueError("campaigns have incompatible datasets or steps")
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    steps = reference["steps"]
    end_tick = int(np.ceil(steps[-1] / 5000) * 5000)
    ticks = np.arange(0, end_tick + 1, 5000)
    tick_labels = ["0" if tick == 0 else f"{tick // 1000}k" for tick in ticks]
    for dataset, (ax, name) in enumerate(zip(axes.flat, reference["names"])):
        for mode, mode_label, linestyle in modes:
            if mode not in campaigns:
                continue
            campaign = campaigns[mode]
            for curves, scale_label, color in (
                (campaign["low"], "$b=1$", SECONDARY),
                (campaign["high"], "$b=10$", PRIMARY),
            ):
                mean, lower, upper = mean_ci(curves[:, dataset])
                ax.plot(
                    steps,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    label=f"{mode_label}, {scale_label}",
                )
                ax.fill_between(
                    steps,
                    lower,
                    upper,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
        ax.set_title(name.replace("_", " "), fontsize=10)
        ax.set_xlim(0, end_tick)
        ax.set_ylim(bottom=0.0)
        ax.set_xticks(ticks, tick_labels)
        ax.tick_params(axis="x", labelbottom=dataset >= 4)
        if dataset >= 4:
            ax.set_xlabel("step")
        if dataset % 4 == 0:
            ax.set_ylabel("nMSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(output_path, dpi=150)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_paired_auc(campaigns: dict[str, dict], output_path: Path):
    """Write one bar-chart panel per mode plus a shared legend, all sharing the
    same y limits so the panels can be typeset as subfigures."""
    modes = (
        (NORMALIZED_MODE, "normalized"),
        (MODE, "lr_adjusted"),
    )
    available_modes = [mode for mode, _ in modes if mode in campaigns]
    if not available_modes:
        raise ValueError("at least one campaign is required")
    display_names = (
        "Electricity",
        "Traffic",
        "Solar",
        "Taxi",
        "Wind Farms",
        "Pedestrian",
        "KDD 2018",
        "FRED-MD",
    )
    x = np.arange(NUM_DATASETS)
    width = 0.38
    bars = (
        ("low", -width / 2, "$b=1$", SECONDARY),
        ("high", width / 2, "$b=10$", PRIMARY),
    )
    top = 0.0
    for mode in available_modes:
        campaign = campaigns[mode]
        for key, _, _, _ in bars:
            mean, _, upper = mean_ci(linear_auc(campaign[key], campaign["steps"]))
            top = max(top, upper.max())
    top *= 1.08
    for mode in available_modes:
        campaign = campaigns[mode]
        fig, ax = plt.subplots(figsize=(5.4, 3.0))
        for key, offset, label, color in bars:
            mean, lower, upper = mean_ci(linear_auc(campaign[key], campaign["steps"]))
            ax.bar(
                x + offset,
                mean,
                width,
                yerr=np.stack((mean - lower, upper - mean)),
                label=label,
                color=color,
                error_kw={"linewidth": 1.0, "capsize": 2.0},
            )
        ax.set_ylim(0.0, top)
        ax.set_ylabel("full-training nMSE AUC")
        ax.set_xticks(x, display_names, rotation=35, ha="right", fontsize=8)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color="0.85", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        save_figure(
            fig, output_path.with_name(f"{output_path.stem}_{dict(modes)[mode]}.png")
        )
    fig = plt.figure(figsize=(5.4, 0.35))
    fig.legend(
        handles=[Patch(color=color, label=label) for _, _, label, color in bars],
        loc="center",
        ncol=len(bars),
        frameon=False,
    )
    save_figure(fig, output_path.with_name(f"{output_path.stem}_shared_legend.png"))


def build_summary(campaign: dict) -> dict:
    low_auc = linear_auc(campaign["low"], campaign["steps"])
    high_auc = linear_auc(campaign["high"], campaign["steps"])
    differences = low_auc - high_auc
    paired_t = ttest_rel(
        low_auc.mean(axis=0),
        high_auc.mean(axis=0),
        alternative="greater",
    )
    tests = [
        wilcoxon(
            differences[:, dataset],
            alternative="greater",
            zero_method="wilcox",
            method="exact",
        )
        for dataset in range(NUM_DATASETS)
    ]
    raw_p_values = np.array([test.pvalue for test in tests])
    adjusted_p_values = holm_adjust(raw_p_values)
    datasets = {}
    for dataset, name in enumerate(campaign["names"]):
        percent_reduction = 100.0 * differences[:, dataset] / low_auc[:, dataset]
        datasets[name] = {
            "auc_b1_mean": float(low_auc[:, dataset].mean()),
            "auc_b10_mean": float(high_auc[:, dataset].mean()),
            "paired_auc_difference_b1_minus_b10": differences[:, dataset].tolist(),
            "mean_auc_difference": float(differences[:, dataset].mean()),
            "median_auc_difference": float(np.median(differences[:, dataset])),
            "median_percent_reduction": float(np.median(percent_reduction)),
            "wilcoxon_w_plus": float(tests[dataset].statistic),
            "wilcoxon_nonzero_pairs": int(np.count_nonzero(differences[:, dataset])),
            "wilcoxon_one_sided_p": float(raw_p_values[dataset]),
            "holm_adjusted_p": float(adjusted_p_values[dataset]),
        }
    return {
        "outcome": "mean ordinary nMSE AUC over the complete trajectory",
        "alternative": "AUC(b=1) - AUC(b=10) > 0",
        "num_complementary_pairs": NUM_PAIRS,
        "general_effect_paired_t": {
            "statistic": float(paired_t.statistic),
            "degrees_of_freedom": int(paired_t.df),
            "one_sided_p": float(paired_t.pvalue),
            "mean_auc_b1": float(low_auc.mean()),
            "mean_auc_b10": float(high_auc.mean()),
            "mean_auc_difference": float(differences.mean()),
        },
        "datasets": datasets,
    }


def main():
    args = parse_args()
    campaigns = {
        MODE: load_campaign(
            args.campaign_root,
            MODE,
            PAIR_ONE_ORIGINAL_PATHS,
        )
    }
    if args.normalized_campaign_root is not None:
        campaigns[NORMALIZED_MODE] = load_campaign(
            args.normalized_campaign_root,
            NORMALIZED_MODE,
            PAIR_ONE_NORMALIZED_PATHS,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_aggregate(
        campaigns[MODE], args.output_dir / "scale_swap_permutation_aggregate.png"
    )
    plot_by_dataset(
        campaigns,
        args.output_dir / "scale_swap_permutation_by_dataset.png",
    )
    plot_paired_auc(
        campaigns, args.output_dir / "scale_swap_permutation_paired_auc.png"
    )
    summary = build_summary(campaigns[MODE])
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

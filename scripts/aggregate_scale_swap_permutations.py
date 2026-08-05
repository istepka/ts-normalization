"""Aggregate the 15-pair LR-adjusted scale-swap permutation campaign."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from scripts.scale_swap_permutation_schedule import (
    HIGH_SCALE,
    LOW_SCALE,
    assignment_pair,
    validate_schedule,
)

MODE = "original_lr_adjusted"
NUM_PAIRS = 15
NUM_DATASETS = 8
T_CRITICAL_95_DF14 = 2.144786688
PAIR_ONE_PATHS = (
    Path("outputs/18985_scale_swap_lr_adjusted_a"),
    Path("outputs/18985_scale_swap_lr_adjusted_b"),
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
    return parser.parse_args()


def load_metrics(path: Path) -> dict:
    with np.load(path / f"metrics_{MODE}.npz") as data:
        names = [str(value) for value in data["names"]]
        steps = data["step"]
        nmse = data["nmse"]
    if len(names) != NUM_DATASETS:
        raise ValueError(f"expected {NUM_DATASETS} datasets in {path}")
    if not np.isfinite(nmse).all():
        raise ValueError(f"non-finite nMSE in {path}")
    return {"names": names, "steps": steps, "nmse": nmse}


def pair_paths(pair_index: int, campaign_root: Path) -> tuple[Path, Path]:
    if pair_index == 1:
        return PAIR_ONE_PATHS
    pair_tag = f"{pair_index:02d}"
    return (
        campaign_root / f"pair_{pair_tag}_a",
        campaign_root / f"pair_{pair_tag}_b",
    )


def load_campaign(campaign_root: Path) -> dict:
    validate_schedule()
    low_curves = []
    high_curves = []
    reference_names = None
    reference_steps = None
    for pair_index in range(1, NUM_PAIRS + 1):
        paths = pair_paths(pair_index, campaign_root)
        arms = [load_metrics(path) for path in paths]
        if reference_names is None:
            reference_names = arms[0]["names"]
            reference_steps = arms[0]["steps"]
        for path, arm in zip(paths, arms):
            if arm["names"] != reference_names or not np.array_equal(
                arm["steps"], reference_steps
            ):
                raise ValueError(f"incompatible metrics in {path}")

        assignment_a, assignment_b = assignment_pair(pair_index)
        assignments = (assignment_a, assignment_b)
        pair_low = np.empty((NUM_DATASETS, len(reference_steps)))
        pair_high = np.empty_like(pair_low)
        for dataset in range(NUM_DATASETS):
            for arm, assignment in zip(arms, assignments):
                curve = arm["nmse"][:, :, dataset].mean(axis=0)
                if assignment[dataset] == LOW_SCALE:
                    pair_low[dataset] = curve
                elif assignment[dataset] == HIGH_SCALE:
                    pair_high[dataset] = curve
                else:
                    raise ValueError("assignment contains an unexpected scale")
        low_curves.append(pair_low)
        high_curves.append(pair_high)
    return {
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


def mean_ci(values: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = values.mean(axis=0)
    half = T_CRITICAL_95_DF14 * values.std(axis=0, ddof=1) / np.sqrt(NUM_PAIRS)
    return mean, mean - half, mean + half


def save_figure(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_aggregate(campaign: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    steps = campaign["steps"]
    for curves, label, color in (
        (campaign["low"], "assigned $b=1$", "tab:blue"),
        (campaign["high"], "assigned $b=10$", "tab:orange"),
    ):
        dataset_curves = curves.mean(axis=0)
        mean = dataset_curves.mean(axis=0)
        half = 2.364624251 * dataset_curves.std(axis=0, ddof=1) / np.sqrt(NUM_DATASETS)
        ax.plot(steps, mean, color=color, label=label)
        ax.fill_between(steps, mean - half, mean + half, color=color, alpha=0.2)
    ax.set_xlabel("step")
    ax.set_ylabel("nMSE")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    save_figure(fig, output_path)


def plot_by_dataset(campaign: dict, output_path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    steps = campaign["steps"]
    end_tick = int(np.ceil(steps[-1] / 5000) * 5000)
    ticks = np.arange(0, end_tick + 1, 5000)
    tick_labels = ["0" if tick == 0 else f"{tick // 1000}k" for tick in ticks]
    for dataset, (ax, name) in enumerate(zip(axes.flat, campaign["names"])):
        for curves, label, color in (
            (campaign["low"], "$b=1$", "tab:blue"),
            (campaign["high"], "$b=10$", "tab:orange"),
        ):
            mean, lower, upper = mean_ci(curves[:, dataset])
            ax.plot(steps, mean, color=color, label=label)
            ax.fill_between(steps, lower, upper, color=color, alpha=0.2)
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


def plot_paired_auc(campaign: dict, output_path: Path):
    low_auc = linear_auc(campaign["low"], campaign["steps"])
    high_auc = linear_auc(campaign["high"], campaign["steps"])
    colors = plt.get_cmap("tab10").colors
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
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    for dataset in range(NUM_DATASETS):
        ax.plot(
            [0, 1],
            [low_auc[:, dataset].mean(), high_auc[:, dataset].mean()],
            color=colors[dataset],
            marker="o",
            linewidth=1.5,
            label=display_names[dataset],
        )
    ax.set_xticks([0, 1], ["$b=1$", "$b=10$"])
    ax.set_ylabel("full-training nMSE AUC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.legend(
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
        ncol=1,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.79, 1.0))
    fig.savefig(output_path, dpi=150)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


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
    campaign = load_campaign(args.campaign_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_aggregate(campaign, args.output_dir / "scale_swap_permutation_aggregate.png")
    plot_by_dataset(campaign, args.output_dir / "scale_swap_permutation_by_dataset.png")
    plot_paired_auc(campaign, args.output_dir / "scale_swap_permutation_paired_auc.png")
    summary = build_summary(campaign)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

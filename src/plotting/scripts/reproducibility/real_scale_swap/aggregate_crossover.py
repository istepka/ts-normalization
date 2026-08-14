"""Analyze the paired eight-dataset scale-swap crossover experiment.

Paper: results.tex "Controlled scale assignment on real-world datasets"
(Fig. loss-space-scale-swap-main) and apd_loss_space_scale_swap.tex
(Table loss-space-scale-swap, Fig. loss-space-scale-swap-effect).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.plotting.core import PRIMARY, SECONDARY, TEAL, mean_ci
from src.plotting.core import save_figure as save_base_figure

BASE_MODES = ["normalized", "original"]
LR_ADJUSTED_MODE = "original_lr_adjusted"
MODE_TITLES = {
    "normalized": "Normalized-space loss",
    "original": "Original-space loss",
    LR_ADJUSTED_MODE: "Original-space loss, LR / 50.5",
}
MODE_AXIS_LABELS = {
    "normalized": "normalized",
    "original": "original",
    LR_ADJUSTED_MODE: "original\nLR / 50.5",
}
LOW_SCALE = 1.0
HIGH_SCALE = 10.0
ASSIGNMENT_A = np.array([LOW_SCALE] * 4 + [HIGH_SCALE] * 4)
ASSIGNMENT_B = np.array([HIGH_SCALE] * 4 + [LOW_SCALE] * 4)
EARLY_END_STEP = 2000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment-a", type=Path, required=True)
    parser.add_argument("--assignment-b", type=Path, required=True)
    parser.add_argument("--lr-adjusted-a", type=Path)
    parser.add_argument("--lr-adjusted-b", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_assignment(path: Path, modes: list[str]) -> dict:
    metrics = {}
    reference_names = None
    reference_steps = None
    for mode in modes:
        with np.load(path / f"metrics_{mode}.npz") as data:
            names = [str(value) for value in data["names"]]
            steps = data["step"]
            if reference_names is None:
                reference_names = names
                reference_steps = steps
            if names != reference_names or not np.array_equal(steps, reference_steps):
                raise ValueError(f"incompatible metrics in {path}")
            metrics[mode] = {
                "nmse": data["nmse"],
                "grad_mag": data["grad_mag"],
            }
            for key, values in metrics[mode].items():
                if not np.isfinite(values).all():
                    raise ValueError(f"non-finite {mode} {key} in {path}")
    if len(reference_names) != len(ASSIGNMENT_A):
        raise ValueError("scale-swap analysis requires exactly eight datasets")
    return {
        "path": str(path),
        "names": reference_names,
        "steps": reference_steps,
        "metrics": metrics,
    }


def validate_assignments(a: dict, b: dict, modes: list[str]):
    if a["names"] != b["names"] or not np.array_equal(a["steps"], b["steps"]):
        raise ValueError("assignments A and B have incompatible datasets or steps")
    for mode in modes:
        if a["metrics"][mode]["nmse"].shape != b["metrics"][mode]["nmse"].shape:
            raise ValueError(f"assignments A and B have incompatible {mode} metrics")


def paired_values(a: dict, b: dict, mode: str, key: str) -> tuple[np.ndarray, ...]:
    low = []
    high = []
    for category, scale_a in enumerate(ASSIGNMENT_A):
        if scale_a == LOW_SCALE:
            low.append(a["metrics"][mode][key][..., category])
            high.append(b["metrics"][mode][key][..., category])
        else:
            low.append(b["metrics"][mode][key][..., category])
            high.append(a["metrics"][mode][key][..., category])
    return np.stack(low), np.stack(high)


def early_log_auc(values: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Seed-average log10-nMSE AUC through step 2,000 for each dataset."""
    keep = steps <= EARLY_END_STEP
    selected_steps = steps[keep]
    if selected_steps[-1] != EARLY_END_STEP:
        raise ValueError(f"metrics must include step {EARLY_END_STEP}")
    log_curve = np.log10(values[:, :, keep]).mean(axis=1)
    return np.trapezoid(log_curve, selected_steps, axis=1) / EARLY_END_STEP


def full_log_auc(values: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Seed-average log10-nMSE AUC over the complete training trajectory."""
    log_curve = np.log10(values).mean(axis=1)
    duration = steps[-1] - steps[0]
    return np.trapezoid(log_curve, steps, axis=1) / duration


def full_linear_auc(values: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Seed-average nMSE AUC over the complete training trajectory."""
    mean_curve = values.mean(axis=1)
    duration = steps[-1] - steps[0]
    return np.trapezoid(mean_curve, steps, axis=1) / duration


def scalar_ci(values: np.ndarray) -> dict[str, float]:
    mean, lower, upper = mean_ci(values)
    return {
        "mean": float(mean),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def log_scalar_ci(values: np.ndarray) -> dict[str, float]:
    mean, lower, upper = mean_ci(np.log10(values))
    return {
        "geometric_mean": float(10**mean),
        "ci95_lower": float(10**lower),
        "ci95_upper": float(10**upper),
    }


def save_figure(fig, path: Path, rect=None):
    save_base_figure(fig, path, dpi=150, tight_layout=True, rect=rect, bbox_inches=None)


def plot_curves(
    a: dict,
    b: dict,
    modes: list[str],
    output_path: Path,
    end_step: int | None,
    yscale: str = "log",
):
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4.5), sharey=True)
    keep = np.ones_like(a["steps"], dtype=bool)
    if end_step is not None:
        keep = a["steps"] <= end_step
    steps = a["steps"][keep]
    for ax, mode in zip(axes, modes):
        low, high = paired_values(a, b, mode, "nmse")
        for values, label in ((low, "assigned b=1"), (high, "assigned b=10")):
            dataset_curves = np.log10(values[:, :, keep]).mean(axis=1)
            mean, lower, upper = mean_ci(dataset_curves)
            (line,) = ax.plot(steps, 10**mean, label=label)
            ax.fill_between(
                steps,
                10**lower,
                10**upper,
                color=line.get_color(),
                alpha=0.25,
                linewidth=0,
            )
        ax.set_yscale(yscale)
        if yscale == "linear":
            ax.set_ylim(bottom=0.0)
        ax.set_xlabel("step")
        ax.set_title(MODE_TITLES[mode])
        ax.legend()
    axes[0].set_ylabel("nMSE")
    title = (
        "Full 30,000-step convergence"
        if end_step is None
        else f"Early convergence (0-{end_step:,} steps)"
    )
    fig.suptitle(title)
    save_figure(fig, output_path)


def plot_mode_by_dataset(
    a: dict,
    b: dict,
    mode: str,
    output_path: Path,
    yscale: str = "log",
):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    steps = a["steps"]
    scale_colors = {LOW_SCALE: SECONDARY, HIGH_SCALE: PRIMARY}
    assignments = (
        (a, ASSIGNMENT_A, "-"),
        (b, ASSIGNMENT_B, "--"),
    )
    end_tick = int(np.ceil(steps[-1] / 5000) * 5000)
    ticks = np.arange(0, end_tick + 1, 5000)
    tick_labels = ["0" if tick == 0 else f"{tick // 1000}k" for tick in ticks]
    for dataset, (ax, name) in enumerate(zip(axes.flat, a["names"])):
        for assignment, scales, linestyle in assignments:
            scale = scales[dataset]
            values = assignment["metrics"][mode]["nmse"][..., dataset]
            seed_curves = np.log10(values)
            mean = seed_curves.mean(axis=0)
            std = seed_curves.std(axis=0, ddof=1)
            color = scale_colors[scale]
            ax.plot(steps, 10**mean, color=color, linestyle=linestyle)
            ax.fill_between(
                steps,
                10 ** (mean - std),
                10 ** (mean + std),
                color=color,
                alpha=0.2,
                linewidth=0,
            )
        ax.set_yscale(yscale)
        if yscale == "linear":
            ax.set_ylim(bottom=0.0)
        ax.set_title(name.replace("_", " "), fontsize=10)
        ax.set_xlim(0, end_tick)
        ax.set_xticks(ticks, tick_labels)
        ax.tick_params(axis="x", labelbottom=dataset >= 4)
        if dataset >= 4:
            ax.set_xlabel("step")
        if dataset % 4 == 0:
            ax.set_ylabel("nMSE")
    legend_handles = [
        Line2D([0], [0], color=scale_colors[LOW_SCALE], label="Scale $b=1$"),
        Line2D([0], [0], color=scale_colors[HIGH_SCALE], label="Scale $b=10$"),
        Line2D([0], [0], color="0.25", linestyle="-", label="Assignment A"),
        Line2D([0], [0], color="0.25", linestyle="--", label="Assignment B"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        frameon=False,
    )
    save_figure(fig, output_path, rect=(0.0, 0.07, 1.0, 1.0))


def plot_paired_effect(a: dict, b: dict, modes: list[str], output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for mode in modes:
        low, high = paired_values(a, b, mode, "nmse")
        dataset_effect = np.log10(low).mean(axis=1) - np.log10(high).mean(axis=1)
        mean, lower, upper = mean_ci(dataset_effect)
        (line,) = ax.plot(a["steps"], mean, label=MODE_TITLES[mode])
        ax.fill_between(
            a["steps"],
            lower,
            upper,
            color=line.get_color(),
            alpha=0.25,
            linewidth=0,
        )
    ax.axhline(0.0, color="0.3", linestyle="--", linewidth=1.2, zorder=0)
    ax.set_xlabel("step")
    ax.set_ylabel("paired log10 nMSE: b=1 minus b=10")
    ax.set_title("Paired scale effect (mean and 95% CI across 8 datasets)")
    ax.legend()
    save_figure(fig, output_path)


def plot_paired_auc(a: dict, b: dict, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 2.7))
    panels = (
        ("normalized", "Normalized-space loss"),
        (LR_ADJUSTED_MODE, "Original-space loss, LR / 50.5"),
    )
    for ax, (mode, title) in zip(axes, panels):
        low, high = paired_values(a, b, mode, "nmse")
        low_auc = full_linear_auc(low, a["steps"])
        high_auc = full_linear_auc(high, a["steps"])
        for low_value, high_value in zip(low_auc, high_auc):
            ax.plot([0, 1], [low_value, high_value], color="0.65", linewidth=1.2)
        ax.scatter(np.zeros(len(low_auc)), low_auc, color=SECONDARY, zorder=2)
        ax.scatter(np.ones(len(high_auc)), high_auc, color=PRIMARY, zorder=2)
        ax.set_xticks([0, 1], ["$b=1$", "$b=10$"])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("full-training nMSE AUC")
    save_figure(fig, output_path)


def plot_gradient_ratios(
    a: dict,
    b: dict,
    modes: list[str],
    output_path: Path,
    yscale: str = "log",
):
    ratios = []
    for mode in modes:
        low, high = paired_values(a, b, mode, "grad_mag")
        ratios.append(10 ** np.log10(high[:, :, 0] / low[:, :, 0]).mean(axis=1))
    ratios = np.stack(ratios, axis=1)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(modes))
    log_mean, log_lower, log_upper = mean_ci(np.log10(ratios))
    mean = 10**log_mean
    error = np.stack((mean - 10**log_lower, 10**log_upper - mean))
    bars = ax.bar(
        x,
        mean,
        yerr=error,
        color=[SECONDARY, PRIMARY, TEAL][: len(modes)],
        capsize=4,
    )
    ax.bar_label(bars, labels=[f"{value:.0f}x" for value in mean], padding=4)
    ax.axhline(1.0, color="0.3", linestyle="--", linewidth=1.2)
    ax.set_yscale(yscale)
    ax.set_xticks(x, [MODE_AXIS_LABELS[mode] for mode in modes])
    ax.set_ylabel("initial gradient ratio: b=10 / b=1")
    ax.set_title("Initial gradient ratio (geometric mean and 95% CI)")
    ax.set_ylim(0.0 if yscale == "linear" else 0.7, 180.0)
    save_figure(fig, output_path)


def build_summary(a: dict, b: dict, modes: list[str]) -> dict:
    summary = {
        "datasets": a["names"],
        "assignment_a": dict(zip(a["names"], ASSIGNMENT_A.tolist())),
        "assignment_b": dict(zip(a["names"], ASSIGNMENT_B.tolist())),
        "auc_definitions": {
            "early": "mean log10(nMSE) over steps 0 through 2000",
            "full": "mean log10(nMSE) over the complete training trajectory",
        },
        "confidence_interval": "paired 95% Student-t interval across 8 datasets",
        "modes": {},
    }
    for mode in modes:
        low_nmse, high_nmse = paired_values(a, b, mode, "nmse")
        auc_delta = early_log_auc(low_nmse, a["steps"]) - early_log_auc(
            high_nmse, a["steps"]
        )
        full_auc_delta = full_log_auc(low_nmse, a["steps"]) - full_log_auc(
            high_nmse, a["steps"]
        )
        final_effect = np.log10(low_nmse[:, :, -1]).mean(axis=1) - np.log10(
            high_nmse[:, :, -1]
        ).mean(axis=1)
        low_grad, high_grad = paired_values(a, b, mode, "grad_mag")
        gradient_ratio = 10 ** np.log10(high_grad[:, :, 0] / low_grad[:, :, 0]).mean(
            axis=1
        )
        summary["modes"][mode] = {
            "paired_auc_low_minus_high": {
                "by_dataset": dict(zip(a["names"], auc_delta.tolist())),
                "mean_ci95": scalar_ci(auc_delta),
            },
            "paired_full_auc_low_minus_high": {
                "by_dataset": dict(zip(a["names"], full_auc_delta.tolist())),
                "mean_ci95": scalar_ci(full_auc_delta),
            },
            "paired_final_log_nmse_low_minus_high": {
                "by_dataset": dict(zip(a["names"], final_effect.tolist())),
                "mean_ci95": scalar_ci(final_effect),
            },
            "init_gradient_ratio_high_over_low": {
                "by_dataset": dict(zip(a["names"], gradient_ratio.tolist())),
                "geometric_mean_ci95": log_scalar_ci(gradient_ratio),
            },
        }
    return summary


def main():
    args = parse_args()
    assignment_a = load_assignment(args.assignment_a, BASE_MODES)
    assignment_b = load_assignment(args.assignment_b, BASE_MODES)
    validate_assignments(assignment_a, assignment_b, BASE_MODES)
    modes = list(BASE_MODES)
    has_adjusted_a = args.lr_adjusted_a is not None
    has_adjusted_b = args.lr_adjusted_b is not None
    if has_adjusted_a != has_adjusted_b:
        raise ValueError("both LR-adjusted assignment paths must be provided together")
    if has_adjusted_a:
        adjusted_a = load_assignment(args.lr_adjusted_a, [LR_ADJUSTED_MODE])
        adjusted_b = load_assignment(args.lr_adjusted_b, [LR_ADJUSTED_MODE])
        validate_assignments(adjusted_a, adjusted_b, [LR_ADJUSTED_MODE])
        for base, adjusted in (
            (assignment_a, adjusted_a),
            (assignment_b, adjusted_b),
        ):
            if base["names"] != adjusted["names"] or not np.array_equal(
                base["steps"], adjusted["steps"]
            ):
                raise ValueError(
                    "LR-adjusted metrics are incompatible with base metrics"
                )
            base["metrics"][LR_ADJUSTED_MODE] = adjusted["metrics"][LR_ADJUSTED_MODE]
        modes.append(LR_ADJUSTED_MODE)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(
        assignment_a,
        assignment_b,
        modes,
        args.output_dir / "scale_swap_curves_full.png",
        end_step=None,
    )
    plot_curves(
        assignment_a,
        assignment_b,
        modes,
        args.output_dir / "scale_swap_curves_early_2000.png",
        end_step=EARLY_END_STEP,
    )
    plot_curves(
        assignment_a,
        assignment_b,
        modes,
        args.output_dir / "scale_swap_curves_full_linear.png",
        end_step=None,
        yscale="linear",
    )
    plot_curves(
        assignment_a,
        assignment_b,
        modes,
        args.output_dir / "scale_swap_curves_early_2000_linear.png",
        end_step=EARLY_END_STEP,
        yscale="linear",
    )
    plot_mode_by_dataset(
        assignment_a,
        assignment_b,
        "original",
        args.output_dir / "scale_swap_original_by_dataset.png",
    )
    plot_mode_by_dataset(
        assignment_a,
        assignment_b,
        "original",
        args.output_dir / "scale_swap_original_by_dataset_linear.png",
        yscale="linear",
    )
    if LR_ADJUSTED_MODE in modes:
        plot_mode_by_dataset(
            assignment_a,
            assignment_b,
            LR_ADJUSTED_MODE,
            args.output_dir / "scale_swap_lr_adjusted_by_dataset.png",
        )
        plot_mode_by_dataset(
            assignment_a,
            assignment_b,
            LR_ADJUSTED_MODE,
            args.output_dir / "scale_swap_lr_adjusted_by_dataset_linear.png",
            yscale="linear",
        )
    plot_paired_effect(
        assignment_a,
        assignment_b,
        modes,
        args.output_dir / "scale_swap_paired_effect.png",
    )
    plot_paired_auc(
        assignment_a,
        assignment_b,
        args.output_dir / "scale_swap_paired_auc.png",
    )
    plot_gradient_ratios(
        assignment_a,
        assignment_b,
        modes,
        args.output_dir / "scale_swap_gradients.png",
    )
    plot_gradient_ratios(
        assignment_a,
        assignment_b,
        modes,
        args.output_dir / "scale_swap_gradients_linear.png",
        yscale="linear",
    )
    summary = build_summary(assignment_a, assignment_b, modes)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

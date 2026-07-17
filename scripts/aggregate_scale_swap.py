"""Analyze the paired eight-dataset scale-swap crossover experiment."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODES = ["normalized", "original"]
LOW_SCALE = 1.0
HIGH_SCALE = 10.0
ASSIGNMENT_A = np.array([LOW_SCALE] * 4 + [HIGH_SCALE] * 4)
ASSIGNMENT_B = np.array([HIGH_SCALE] * 4 + [LOW_SCALE] * 4)
EARLY_END_STEP = 2000
T_CRITICAL_95_DF7 = 2.364624251


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment-a", type=Path, required=True)
    parser.add_argument("--assignment-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_assignment(path: Path) -> dict:
    metrics = {}
    reference_names = None
    reference_steps = None
    for mode in MODES:
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


def validate_assignments(a: dict, b: dict):
    if a["names"] != b["names"] or not np.array_equal(a["steps"], b["steps"]):
        raise ValueError("assignments A and B have incompatible datasets or steps")
    for mode in MODES:
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


def mean_ci(values: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = values.mean(axis=0)
    half = T_CRITICAL_95_DF7 * values.std(axis=0, ddof=1) / np.sqrt(len(values))
    return mean, mean - half, mean + half


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


def save_figure(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_curves(a: dict, b: dict, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    keep = a["steps"] <= EARLY_END_STEP
    steps = a["steps"][keep]
    for ax, mode in zip(axes, MODES):
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
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_title(f"{mode.capitalize()}-space loss")
        ax.legend()
    axes[0].set_ylabel("nMSE")
    save_figure(fig, output_path)


def plot_paired_auc(a: dict, b: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(MODES))
    deltas = []
    for mode in MODES:
        low, high = paired_values(a, b, mode, "nmse")
        deltas.append(early_log_auc(low, a["steps"]) - early_log_auc(high, a["steps"]))
    deltas = np.stack(deltas, axis=1)
    for dataset in range(len(a["names"])):
        ax.plot(x, deltas[dataset], color="0.75", linewidth=1.0, zorder=1)
        ax.scatter(x, deltas[dataset], s=28, zorder=2)
    mean, lower, upper = mean_ci(deltas)
    ax.errorbar(
        x,
        mean,
        yerr=np.stack((mean - lower, upper - mean)),
        color="black",
        marker="D",
        capsize=4,
        linestyle="none",
        label="mean and 95% CI",
        zorder=3,
    )
    ax.axhline(0.0, color="0.3", linestyle="--", linewidth=1.2)
    ax.set_xticks(x, ["normalized", "original"])
    ax.set_ylabel("early log-nMSE AUC: b=1 minus b=10")
    ax.set_title("Paired scale-swap effect by dataset")
    ax.legend()
    save_figure(fig, output_path)


def plot_gradient_ratios(a: dict, b: dict, output_path: Path):
    ratios = []
    for mode in MODES:
        low, high = paired_values(a, b, mode, "grad_mag")
        ratios.append(10 ** np.log10(high[:, :, 0] / low[:, :, 0]).mean(axis=1))
    ratios = np.stack(ratios, axis=1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(MODES))
    for dataset in range(len(a["names"])):
        ax.plot(x, ratios[dataset], color="0.75", linewidth=1.0, zorder=1)
        ax.scatter(x, ratios[dataset], s=28, zorder=2)
    log_mean, log_lower, log_upper = mean_ci(np.log10(ratios))
    mean = 10**log_mean
    ax.errorbar(
        x,
        mean,
        yerr=np.stack((mean - 10**log_lower, 10**log_upper - mean)),
        color="black",
        marker="D",
        capsize=4,
        linestyle="none",
        label="geometric mean and 95% CI",
        zorder=3,
    )
    ax.axhline(1.0, color="0.3", linestyle="--", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xticks(x, ["normalized", "original"])
    ax.set_ylabel("initial gradient ratio: b=10 / b=1")
    ax.set_title("Assigned-scale gradient effect by dataset")
    ax.legend()
    save_figure(fig, output_path)


def build_summary(a: dict, b: dict) -> dict:
    summary = {
        "datasets": a["names"],
        "assignment_a": dict(zip(a["names"], ASSIGNMENT_A.tolist())),
        "assignment_b": dict(zip(a["names"], ASSIGNMENT_B.tolist())),
        "early_auc": "mean log10(nMSE) over steps 0 through 2000",
        "confidence_interval": "paired 95% Student-t interval across 8 datasets",
        "modes": {},
    }
    for mode in MODES:
        low_nmse, high_nmse = paired_values(a, b, mode, "nmse")
        auc_delta = early_log_auc(low_nmse, a["steps"]) - early_log_auc(
            high_nmse, a["steps"]
        )
        low_grad, high_grad = paired_values(a, b, mode, "grad_mag")
        gradient_ratio = 10 ** np.log10(high_grad[:, :, 0] / low_grad[:, :, 0]).mean(
            axis=1
        )
        summary["modes"][mode] = {
            "paired_auc_low_minus_high": {
                "by_dataset": dict(zip(a["names"], auc_delta.tolist())),
                "mean_ci95": scalar_ci(auc_delta),
            },
            "init_gradient_ratio_high_over_low": {
                "by_dataset": dict(zip(a["names"], gradient_ratio.tolist())),
                "geometric_mean_ci95": log_scalar_ci(gradient_ratio),
            },
        }
    return summary


def main():
    args = parse_args()
    assignment_a = load_assignment(args.assignment_a)
    assignment_b = load_assignment(args.assignment_b)
    validate_assignments(assignment_a, assignment_b)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(assignment_a, assignment_b, args.output_dir / "scale_swap_curves.png")
    plot_paired_auc(
        assignment_a, assignment_b, args.output_dir / "scale_swap_paired_auc.png"
    )
    plot_gradient_ratios(
        assignment_a, assignment_b, args.output_dir / "scale_swap_gradients.png"
    )
    summary = build_summary(assignment_a, assignment_b)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

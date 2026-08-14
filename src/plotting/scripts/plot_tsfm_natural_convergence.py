"""Plot paper-ready natural-scale TSFM convergence figures."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

OUTPUT_DIR = Path("outputs/tsfm_natural_convergence")
EARLY_JOB = 29395
EARLY_STEPS = (1, 10, 50, 100, 250)
COLORS = {"normalized": "#0072B2", "original": "#D55E00"}
LABELS = {"normalized": "Normalized-space loss", "original": "Original-space loss"}

MODELS = {
    "MOMENT": {
        "key": "moment",
        "conditions": {
            "normalized": "moment_normalized",
            "original": "moment_original",
        },
        "histories": {
            condition: [
                Path(
                    f"outputs/gifteval_moment_28827_seed{seed}_moment_"
                    f"{condition}_natural/history.json"
                )
                for seed in range(4)
            ]
            for condition in ("normalized", "original")
        },
    },
    "TimesFM": {
        "key": "timesfm",
        "conditions": {
            "normalized": "timesfm_normalized",
            "original": "timesfm_native_original",
        },
        "histories": {
            "normalized": [
                Path(
                    f"outputs/timesfm_whole_context_natural_"
                    f"{28828 if seed < 2 else 27955}_seed{seed}_"
                    "timesfm_normalized/history.json"
                )
                for seed in range(4)
            ],
            "original": [
                Path(
                    f"outputs/timesfm_whole_context_natural_"
                    f"{28828 if seed < 2 else 27955}_seed{seed}_"
                    "timesfm_native_original/history.json"
                )
                for seed in range(4)
            ],
        },
    },
    "Chronos-2": {
        "key": "chronos2",
        "conditions": {
            "normalized": "chronos2_normalized",
            "original": "chronos2_original",
        },
        "histories": {
            condition: [
                Path(
                    f"outputs/gifteval_chronos2_{27484 if seed < 2 else 27655}_"
                    f"seed{seed}_chronos2_{condition}_natural/history.json"
                )
                for seed in range(4)
            ]
            for condition in ("normalized", "original")
        },
    },
    "MOIRAI 2.0": {
        "key": "moirai2",
        "conditions": {
            "normalized": "moirai2_normalized",
            "original": "moirai2_original",
        },
        "histories": {
            condition: [
                Path(
                    f"outputs/gifteval_moirai2_28035_seed{seed}_moirai2_"
                    f"{condition}_natural/history.json"
                )
                for seed in range(4)
            ]
            for condition in ("normalized", "original")
        },
    },
}


def metric_values(history: dict, metric: str) -> np.ndarray:
    if metric == "global_mean":
        return np.asarray(
            [report["pooled_mase"] for report in history["reports"]], dtype=float
        )
    if metric == "dataset_median":
        return np.asarray(
            [
                np.median(
                    list(report["mase"]["dataset"]["per_source_mean_error"].values())
                )
                for report in history["reports"]
            ],
            dtype=float,
        )
    raise ValueError(f"unknown metric {metric!r}")


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_early(metric: str, ylabel: str, stem: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15), sharey=True)
    for axis, (title, model) in zip(axes, MODELS.items(), strict=True):
        for loss_space in ("normalized", "original"):
            condition = model["conditions"][loss_space]
            values = []
            for step in EARLY_STEPS:
                path = OUTPUT_DIR.parent / (
                    f"loss_early_sanity_{EARLY_JOB}_{model['key']}_{condition}_"
                    f"step{step}/history.json"
                )
                values.append(metric_values(json.loads(path.read_text()), metric)[0])
            axis.plot(
                EARLY_STEPS,
                values,
                color=COLORS[loss_space],
                label=LABELS[loss_space],
                marker="o",
                markersize=2.5,
                linewidth=1.25,
            )
        axis.set_title(title, fontsize=8)
        axis.set_xscale("symlog", linthresh=1)
        axis.set_xlim(0, 250)
        axis.set_xlabel("Training step", fontsize=7)
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.tick_params(labelsize=6)
    axes[0].set_ylabel(ylabel, fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        frameon=False,
        ncol=2,
        fontsize=7,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.78, wspace=0.12)
    save_figure(fig, stem)


def plot_full(metric: str, ylabel: str, stem: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15), sharey=True)
    for axis, (title, model) in zip(axes, MODELS.items(), strict=True):
        for loss_space in ("normalized", "original"):
            histories = [
                json.loads(path.read_text()) for path in model["histories"][loss_space]
            ]
            steps = np.asarray(histories[0]["step"])
            assert all(np.array_equal(history["step"], steps) for history in histories)
            values = np.asarray(
                [metric_values(history, metric) for history in histories]
            )
            mean = values.mean(axis=0)
            interval = t.ppf(0.975, 3) * values.std(axis=0, ddof=1) / 2
            axis.plot(
                steps,
                mean,
                color=COLORS[loss_space],
                label=LABELS[loss_space],
                linewidth=1.35,
            )
            axis.fill_between(
                steps,
                mean - interval,
                mean + interval,
                color=COLORS[loss_space],
                alpha=0.14,
                linewidth=0,
            )
        axis.set_title(title, fontsize=8)
        axis.set_xlim(0, 30_000)
        axis.set_xticks((0, 15_000, 30_000), labels=("0", "15k", "30k"))
        axis.set_xlabel("Training step", fontsize=7)
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.tick_params(labelsize=6)
    axes[0].set_ylabel(ylabel, fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        frameon=False,
        ncol=2,
        fontsize=7,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.78, wspace=0.12)
    save_figure(fig, stem)


def main() -> None:
    plot_early(
        "dataset_median",
        "Median per-dataset MASE",
        "paper_early_median_mase",
    )
    plot_full(
        "dataset_median",
        "Median per-dataset MASE",
        "paper_full_median_mase",
    )
    plot_early(
        "global_mean",
        "Mean MASE across windows",
        "paper_early_mean_mase",
    )
    plot_full(
        "global_mean",
        "Mean MASE across windows",
        "paper_full_mean_mase",
    )


if __name__ == "__main__":
    main()

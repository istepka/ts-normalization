"""Aggregate eight independent variance-bin experiments across datasets.

Each input is dataset_name=output_dir. Seeds are averaged within a dataset
before computing means and 95% Student-t confidence intervals across the eight
datasets, which are the independent statistical units.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SETUP_LABELS = [
    "normalized",
    "original",
    "original_equalvar",
    "original_gradmatch",
]
NUM_DATASETS = 8
T_CRITICAL_95_DF7 = 2.364624251
GLOBAL_STYLE = {"color": "0.35", "linestyle": "--", "linewidth": 1.6}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs=NUM_DATASETS, metavar="NAME=OUTPUT_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_datasets(specs: list[str]) -> tuple[list[dict], list[str]]:
    datasets = []
    reference_names = None
    reference_steps = None
    for spec in specs:
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        metrics = {}
        for label in SETUP_LABELS:
            with np.load(path / f"metrics_{label}.npz") as data:
                names = [str(value) for value in data["names"]]
                steps = data["step"]
                if reference_names is None:
                    reference_names = names
                    reference_steps = steps
                if names != reference_names or not np.array_equal(
                    steps, reference_steps
                ):
                    raise ValueError(f"incompatible metrics in {path}")
                metrics[label] = {
                    "step": steps,
                    "nmse": data["nmse"],
                    "global_nmse": data["global_nmse"],
                    "grad_mag": data["grad_mag"],
                }
        datasets.append({"name": name, "path": str(path), "metrics": metrics})
    return datasets, reference_names


def stack_dataset_means(datasets: list[dict], label: str, key: str) -> np.ndarray:
    return np.stack(
        [dataset["metrics"][label][key].mean(axis=0) for dataset in datasets],
        axis=0,
    )


def mean_ci(values: np.ndarray, log_space: bool) -> tuple[np.ndarray, ...]:
    if values.shape[0] != NUM_DATASETS:
        raise ValueError(f"expected {NUM_DATASETS} datasets")
    work = np.log10(values) if log_space else values
    center = work.mean(axis=0)
    half = T_CRITICAL_95_DF7 * work.std(axis=0, ddof=1) / np.sqrt(NUM_DATASETS)
    if log_space:
        return 10**center, 10 ** (center - half), 10 ** (center + half)
    return center, center - half, center + half


def save_figure(fig, output_path: Path):
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_nmse_panels(
    datasets: list[dict],
    names: list[str],
    labels: list[str],
    titles: list[str],
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    steps = datasets[0]["metrics"][labels[0]]["step"]
    for panel, (ax, label, title) in enumerate(zip(axes, labels, titles)):
        nmse = stack_dataset_means(datasets, label, "nmse")
        mean, lower, upper = mean_ci(nmse, log_space=False)
        for category, name in enumerate(names):
            (line,) = ax.plot(steps, mean[:, category], label=name)
            ax.fill_between(
                steps,
                lower[:, category],
                upper[:, category],
                color=line.get_color(),
                alpha=0.25,
                linewidth=0,
            )
        global_nmse = stack_dataset_means(datasets, label, "global_nmse")
        global_mean, _, _ = mean_ci(global_nmse, log_space=False)
        ax.plot(steps, global_mean, label="global", **GLOBAL_STYLE)
        ax.set_xlim(0, 2000)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("step")
        ax.set_title(title)
        if panel == 0:
            ax.set_ylabel("nMSE")
            ax.legend()
    save_figure(fig, output_path)


def plot_gradients(datasets: list[dict], names: list[str], output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(names))
    width = 0.38
    for offset, label in zip((-width / 2, width / 2), ("normalized", "original")):
        gradients = stack_dataset_means(datasets, label, "grad_mag")[:, 0, :]
        mean, lower, upper = mean_ci(gradients, log_space=True)
        error = np.stack((mean - lower, upper - mean))
        ax.bar(
            x + offset,
            mean,
            width,
            yerr=error,
            capsize=3,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlabel("within-dataset context-variance bin")
    ax.set_ylabel(r"$\|\partial \mathcal{L}/\partial \hat z\|$")
    ax.set_title("Initialization gradients across datasets")
    ax.legend()
    save_figure(fig, output_path)


def scalar_ci(values: np.ndarray) -> dict[str, float]:
    mean, lower, upper = mean_ci(values, log_space=False)
    return {
        "mean": float(mean),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def build_summary(datasets: list[dict], names: list[str]) -> dict:
    summary = {
        "datasets": [
            {"name": dataset["name"], "output_dir": dataset["path"]}
            for dataset in datasets
        ],
        "confidence_interval": (
            "95% Student-t interval across 8 dataset means after averaging seeds"
        ),
        "setups": {},
    }
    for label in SETUP_LABELS:
        final_nmse = stack_dataset_means(datasets, label, "nmse")[:, -1, :]
        final_global = stack_dataset_means(datasets, label, "global_nmse")[:, -1]
        spread = final_nmse.max(axis=1) / final_nmse.min(axis=1)
        summary["setups"][label] = {
            "final_nmse": {
                name: scalar_ci(final_nmse[:, category])
                for category, name in enumerate(names)
            },
            "final_global_nmse": scalar_ci(final_global),
            "nmse_spread_ratio": scalar_ci(spread),
        }

    gradients = stack_dataset_means(datasets, "original", "grad_mag")[:, 0, :]
    ratios = gradients / gradients.min(axis=1, keepdims=True)
    summary["original_init_grad_ratio"] = {
        name: scalar_ci(ratios[:, category]) for category, name in enumerate(names)
    }
    return summary


def main():
    args = parse_args()
    datasets, names = load_datasets(args.datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_nmse_panels(
        datasets,
        names,
        ["normalized", "original"],
        ["Normalized-space loss", "Original-space loss"],
        args.output_dir / "nmse_core_dataset_ci.png",
    )
    plot_nmse_panels(
        datasets,
        names,
        ["original_equalvar", "original_gradmatch"],
        ["Original + equal variance", "Original + grad-norm matched"],
        args.output_dir / "nmse_controls_dataset_ci.png",
    )
    plot_gradients(datasets, names, args.output_dir / "grad_magnitude_dataset_ci.png")
    summary = build_summary(datasets, names)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

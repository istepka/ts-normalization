"""Build the robust TSFM tables reported in the loss-space paper section."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.gifteval.window_index import stable_seed
from src.plotting.core import MODELS, capped_gini, final_dataset_mase, mean_ci

OUTPUT_DIR = Path("outputs/2026-08-13/analysis/tsfm_pretraining/four_model_loss_space")
TIMESFM_CONTROLLED_ROOT = Path("outputs/2026-08-10/experiments/legacy_runs")
CONTROLLED_MODELS = {
    "TimesFM": {
        "paths": {
            condition: {
                assignment: [
                    TIMESFM_CONTROLLED_ROOT
                    / (
                        "timesfm_robust_27140_whole_context_mse_"
                        f"seed{seed}_{run_condition}_{assignment}"
                    )
                    / "history.json"
                    for seed in range(4)
                ]
                for assignment in ("A", "B")
            }
            for condition, run_condition in {
                "normalized": "timesfm_normalized",
                "original": "timesfm_native_original",
            }.items()
        },
    },
    "Chronos-2": {
        "paths": {
            condition: {
                assignment: [
                    Path(
                        "outputs/2026-08-12/experiments/tsfm_pretraining/"
                        "chronos2/gifteval_chronos2_b512_29436"
                    )
                    / f"seed{seed}_chronos2_{condition}_{assignment}"
                    / "history.json"
                    for seed in range(4)
                ]
                for assignment in ("A", "B")
            }
            for condition in ("normalized", "original")
        },
    },
    "MOIRAI 2.0": {
        "paths": {
            condition: {
                assignment: [
                    Path(
                        "outputs/2026-08-12/experiments/tsfm_pretraining/"
                        "moirai2/gifteval_moirai2_b512_29437"
                    )
                    / f"seed{seed}_moirai2_{condition}_{assignment}"
                    / "history.json"
                    for seed in range(4)
                ]
                for assignment in ("A", "B")
            }
            for condition in ("normalized", "original")
        },
    },
}


def interval(values: list[float]) -> tuple[float, float]:
    """Mean and 95% t-interval half-width of a one-dimensional sample."""
    mean, _, upper = mean_ci(np.asarray(values))
    return float(mean), float(upper - mean)


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    median = float(np.median(values))
    return {
        "median_mase": median,
        "p90_p50": float(np.percentile(values, 90) / median),
        "capped_gini": capped_gini(values),
    }


def natural_table() -> list[dict]:
    rows = []
    for model_name, model in MODELS.items():
        values = {}
        for condition in ("normalized", "original"):
            histories = [
                final_dataset_mase(path) for path in model["histories"][condition]
            ]
            datasets = sorted(histories[0])
            assert all(sorted(history) == datasets for history in histories)
            values[condition] = {
                dataset: float(np.mean([history[dataset] for history in histories]))
                for dataset in datasets
            }
        normalized = np.asarray(list(values["normalized"].values()))
        original = np.asarray(list(values["original"].values()))
        rows.append(
            {
                "model": model_name,
                "task": "Reconstruction" if model_name == "MOMENT" else "Forecasting",
                "n_datasets": len(normalized),
                "normalized": distribution_summary(normalized),
                "original": distribution_summary(original),
                "normalized_wins": int(np.sum(normalized < original)),
            }
        )
    return rows


def scale_groups() -> dict[str, int]:
    index = pd.read_parquet(
        "outputs/gifteval_window_index/context512_pred128.parquet",
        columns=["dataset"],
    )
    datasets = sorted(
        index["dataset"].unique(),
        key=lambda dataset: stable_seed(0, "dataset_scale", dataset),
    )
    split = len(datasets) // 2
    return {dataset: int(i >= split) for i, dataset in enumerate(datasets)}


def controlled_tables() -> tuple[list[dict], list[dict]]:
    groups = scale_groups()
    scale_rows = []
    accuracy_rows = []
    for model_name, model in CONTROLLED_MODELS.items():
        histories = {
            condition: {
                assignment: [
                    final_dataset_mase(path)
                    for path in model["paths"][condition][assignment]
                ]
                for assignment in ("A", "B")
            }
            for condition in ("normalized", "original")
        }
        datasets = sorted(histories["normalized"]["A"][0])
        for condition in ("original", "normalized"):
            seed_ratios = []
            dataset_low = {dataset: [] for dataset in datasets}
            dataset_high = {dataset: [] for dataset in datasets}
            for seed in range(4):
                assignment_a = histories[condition]["A"][seed]
                assignment_b = histories[condition]["B"][seed]
                current_ratios = []
                for dataset in datasets:
                    low, high = (
                        (assignment_a[dataset], assignment_b[dataset])
                        if groups[dataset] == 0
                        else (assignment_b[dataset], assignment_a[dataset])
                    )
                    ratio = low / high
                    current_ratios.append(ratio)
                    dataset_low[dataset].append(low)
                    dataset_high[dataset].append(high)
                seed_ratios.append(float(np.exp(np.mean(np.log(current_ratios)))))
            mean, half_width = interval(seed_ratios)
            scale_rows.append(
                {
                    "model": model_name,
                    "loss_space": condition.capitalize(),
                    "mase_ratio_b1_b10": mean,
                    "ci95_half_width": half_width,
                    "datasets_favoring_b10": int(
                        np.sum(
                            [
                                np.mean(dataset_low[dataset])
                                > np.mean(dataset_high[dataset])
                                for dataset in datasets
                            ]
                        )
                    ),
                    "n_datasets": len(datasets),
                }
            )

        seed_accuracy = []
        dataset_accuracy = {dataset: [] for dataset in datasets}
        inequality = {
            condition: {"p90_p50": [], "capped_gini": []}
            for condition in ("original", "normalized")
        }
        for seed in range(4):
            seed_ratios = []
            for assignment in ("A", "B"):
                normalized = histories["normalized"][assignment][seed]
                original = histories["original"][assignment][seed]
                for dataset in datasets:
                    ratio = normalized[dataset] / original[dataset]
                    seed_ratios.append(ratio)
                    dataset_accuracy[dataset].append(ratio)
                for condition in ("original", "normalized"):
                    values = np.asarray(
                        list(histories[condition][assignment][seed].values())
                    )
                    summary = distribution_summary(values)
                    inequality[condition]["p90_p50"].append(summary["p90_p50"])
                    inequality[condition]["capped_gini"].append(summary["capped_gini"])
            seed_accuracy.append(float(np.exp(np.mean(np.log(seed_ratios)))))
        inequality_differences = {}
        for metric in ("p90_p50", "capped_gini"):
            original = np.asarray(inequality["original"][metric]).reshape(4, 2)
            normalized = np.asarray(inequality["normalized"][metric]).reshape(4, 2)
            difference, difference_half_width = interval(
                list(normalized.mean(axis=1) - original.mean(axis=1))
            )
            inequality_differences[metric] = {
                "difference": difference,
                "ci95_half_width": difference_half_width,
            }
        mean, half_width = interval(seed_accuracy)
        accuracy_rows.append(
            {
                "model": model_name,
                "normalized_original_mase_ratio": mean,
                "ci95_half_width": half_width,
                "normalized_wins": int(
                    np.sum(
                        [
                            np.exp(np.mean(np.log(dataset_accuracy[dataset]))) < 1
                            for dataset in datasets
                        ]
                    )
                ),
                "n_datasets": len(datasets),
                "original_p90_p50": float(np.mean(inequality["original"]["p90_p50"])),
                "normalized_p90_p50": float(
                    np.mean(inequality["normalized"]["p90_p50"])
                ),
                "p90_p50_difference": inequality_differences["p90_p50"]["difference"],
                "p90_p50_difference_ci95_half_width": inequality_differences["p90_p50"][
                    "ci95_half_width"
                ],
                "original_capped_gini": float(
                    np.mean(inequality["original"]["capped_gini"])
                ),
                "normalized_capped_gini": float(
                    np.mean(inequality["normalized"]["capped_gini"])
                ),
                "capped_gini_difference": inequality_differences["capped_gini"][
                    "difference"
                ],
                "capped_gini_difference_ci95_half_width": inequality_differences[
                    "capped_gini"
                ]["ci95_half_width"],
            }
        )
    return scale_rows, accuracy_rows


def main() -> None:
    natural = natural_table()
    scale, accuracy = controlled_tables()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "natural_dataset_scales": natural,
        "controlled_scale_effect": scale,
        "controlled_accuracy_and_inequality": accuracy,
    }
    (OUTPUT_DIR / "paper_tables.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.json_normalize(natural).to_csv(
        OUTPUT_DIR / "natural_dataset_scales.csv", index=False
    )
    pd.DataFrame(scale).to_csv(OUTPUT_DIR / "controlled_scale_effect.csv", index=False)
    pd.DataFrame(accuracy).to_csv(
        OUTPUT_DIR / "controlled_accuracy_and_inequality.csv", index=False
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

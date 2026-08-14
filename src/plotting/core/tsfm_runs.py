"""Run registry and history readers for the four-model TSFM figures."""

import json
from pathlib import Path

import numpy as np

from src.plotting.core.palette import PRIMARY, SECONDARY

LEGACY_ROOT = Path("outputs/2026-08-11/experiments/legacy_runs")
TIMESFM_ROOT = Path(
    "outputs/2026-08-13/experiments/tsfm_pretraining/timesfm/"
    "timesfm_natural_eval250_30192"
)
CHRONOS_ROOT = Path(
    "outputs/2026-08-12/experiments/tsfm_pretraining/chronos2/"
    "gifteval_chronos2_b512_29436"
)
MOIRAI_ROOT = Path(
    "outputs/2026-08-12/experiments/tsfm_pretraining/moirai2/"
    "gifteval_moirai2_b512_29437"
)
COLORS = {"normalized": SECONDARY, "original": PRIMARY}
LABELS = {"normalized": "Normalized-space loss", "original": "Original-space loss"}
LOSS_SPACES = ("normalized", "original")

MODELS = {
    "MOMENT": {
        "key": "moment",
        "conditions": {
            "normalized": "moment_normalized",
            "original": "moment_original",
        },
        "histories": {
            condition: [
                LEGACY_ROOT
                / f"gifteval_moment_28827_seed{seed}_moment_{condition}_natural"
                / "history.json"
                for seed in range(4)
            ]
            for condition in LOSS_SPACES
        },
    },
    "TimesFM": {
        "key": "timesfm",
        "conditions": {
            "normalized": "timesfm_normalized",
            "original": "timesfm_native_original",
        },
        "histories": {
            loss_space: [
                TIMESFM_ROOT
                / f"seed{seed}_timesfm_{run_condition}_natural"
                / "history.json"
                for seed in range(4)
            ]
            for loss_space, run_condition in {
                "normalized": "normalized",
                "original": "native_original",
            }.items()
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
                CHRONOS_ROOT
                / f"seed{seed}_chronos2_{condition}_natural"
                / "history.json"
                for seed in range(4)
            ]
            for condition in LOSS_SPACES
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
                MOIRAI_ROOT / f"seed{seed}_moirai2_{condition}_natural" / "history.json"
                for seed in range(4)
            ]
            for condition in LOSS_SPACES
        },
    },
}


def load_histories(paths: list[Path], end_step: int | None = None) -> list[dict]:
    """Read seed histories that share one evaluation schedule."""
    histories = [json.loads(path.read_text()) for path in paths]
    steps = histories[0]["step"]
    assert all(history["step"] == steps for history in histories)
    if end_step is not None:
        assert all(history["step"][-1] == end_step for history in histories)
    return histories


def final_dataset_mase(path: Path, end_step: int = 30_000) -> dict[str, float]:
    history = json.loads(path.read_text())
    assert history["step"][-1] == end_step
    return history["reports"][-1]["mase"]["dataset"]["per_source_mean_error"]


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


def seed_metric_curves(paths: list[Path], metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Evaluation steps and the per-seed metric curves stacked seed-first."""
    histories = load_histories(paths)
    steps = np.asarray(histories[0]["step"])
    values = np.asarray([metric_values(history, metric) for history in histories])
    return steps, values


def per_dataset_curves(
    histories: list[dict],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Steps, dataset names, and per-seed per-dataset MASE curves."""
    steps = np.asarray(histories[0]["step"])
    datasets = sorted(
        histories[0]["reports"][0]["mase"]["dataset"]["per_source_mean_error"]
    )
    values = np.asarray(
        [
            [
                [
                    report["mase"]["dataset"]["per_source_mean_error"][dataset]
                    for dataset in datasets
                ]
                for report in history["reports"]
            ]
            for history in histories
        ],
        dtype=float,
    )
    return steps, datasets, values


def capped_gini(values: np.ndarray) -> float:
    """Gini of per-dataset errors after capping at the 90th percentile."""
    cap = np.percentile(values, 90)
    capped = np.sort(np.minimum(values, cap))
    ranks = np.arange(1, capped.size + 1)
    return float(
        np.sum((2 * ranks - capped.size - 1) * capped) / (capped.size * capped.sum())
    )

"""Analyze replicated TimesFM controlled-scale runs.

The trained run is the replication unit. Dataset errors from one shared model
are averaged within seed before uncertainty is computed across seeds.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats

CONDITION_LABELS = {
    "timesfm_native_original": "original",
    "timesfm_normalized": "normalized",
}
METRICS = ("final_mase", "mase_auc")
INEQUALITY_METRICS = ("mase_gini", "mase_iqr")


def linear_auc(steps: np.ndarray, values: np.ndarray) -> float:
    if steps.size < 2:
        raise ValueError("at least two evaluation checkpoints are required")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("MASE trajectory must be finite and nonnegative")
    duration = steps[-1] - steps[0]
    if duration <= 0:
        raise ValueError("evaluation steps must be strictly increasing")
    return float(np.trapezoid(values, steps) / duration)


def find_run_dirs(outputs_dir: Path, jobtag: str) -> list[Path]:
    return sorted(
        run_dir
        for run_dir in outputs_dir.glob(f"{jobtag}_*")
        if run_dir.is_dir() and run_dir.name != f"{jobtag}_analysis"
    )


def read_runs(outputs_dir: Path, jobtag: str) -> pd.DataFrame:
    rows = []
    run_dirs = find_run_dirs(outputs_dir, jobtag)
    if not run_dirs:
        raise FileNotFoundError(f"no run directories found for {jobtag}")

    for run_dir in run_dirs:
        cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
        history = json.loads((run_dir / "history.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        index_meta = json.loads((run_dir / "window_index_meta.json").read_text())
        if summary["optimization"]["steps_skipped"] != 0:
            raise ValueError(f"{run_dir} skipped optimizer steps")
        experiment_kind = str(cfg.experiment_kind)
        if experiment_kind not in {"controlled_scale", "natural_mixture"}:
            raise ValueError(f"unsupported experiment kind in {run_dir}")

        condition = CONDITION_LABELS[str(cfg.condition)]
        assignment = (
            str(cfg.scale_assignment)
            if experiment_kind == "controlled_scale"
            else "natural"
        )
        steps = np.asarray(history["step"], dtype=np.int64)
        reports = history["reports"]
        datasets = sorted(reports[-1]["mase"]["dataset"]["per_source_mean_error"])
        groups = index_meta["dataset_scale_groups"]

        for dataset in datasets:
            values = np.asarray(
                [
                    report["mase"]["dataset"]["per_source_mean_error"][dataset]
                    for report in reports
                ],
                dtype=np.float64,
            )
            if experiment_kind == "controlled_scale":
                group = int(groups[dataset])
                low_group = 0 if assignment == "A" else 1
                scale = cfg.scale_b_low if group == low_group else cfg.scale_b_high
            else:
                scale = 1.0
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "experiment_kind": experiment_kind,
                    "seed": int(cfg.seed),
                    "normalization_mode": str(cfg.timesfm.normalization_mode),
                    "objective": str(cfg.timesfm.objective),
                    "condition": condition,
                    "assignment": assignment,
                    "dataset": dataset,
                    "scale": float(scale),
                    "final_mase": float(values[-1]),
                    "mase_auc": linear_auc(steps, values),
                }
            )
    return pd.DataFrame(rows)


def pair_scale_effects(rows: pd.DataFrame) -> pd.DataFrame:
    index = ["seed", "normalization_mode", "objective", "condition", "dataset"]
    paired = rows.pivot(index=index, columns="scale", values=list(METRICS))
    if paired.isna().any().any():
        raise ValueError("missing complementary scale result")
    scales = sorted(rows["scale"].unique())
    if len(scales) != 2:
        raise ValueError(f"expected two controlled scales, found {scales}")

    out = paired.index.to_frame(index=False)
    for metric in METRICS:
        out[f"{metric}_scale_effect"] = (
            paired[(metric, scales[0])].to_numpy()
            - paired[(metric, scales[1])].to_numpy()
        )
    return out


def difference_in_differences(effects: pd.DataFrame) -> pd.DataFrame:
    index = ["seed", "normalization_mode", "objective", "dataset"]
    pivot = effects.pivot(index=index, columns="condition")
    required = {"original", "normalized"}
    if set(pivot.columns.get_level_values("condition")) != required:
        raise ValueError("both original and normalized conditions are required")

    out = pivot.index.to_frame(index=False)
    for metric in METRICS:
        column = f"{metric}_scale_effect"
        out[f"{metric}_did"] = (
            pivot[(column, "original")].to_numpy()
            - pivot[(column, "normalized")].to_numpy()
        )
    return out


def condition_effects(rows: pd.DataFrame) -> pd.DataFrame:
    index = [
        "seed",
        "normalization_mode",
        "objective",
        "assignment",
        "dataset",
        "scale",
    ]
    pivot = rows.pivot(index=index, columns="condition")
    required = {"original", "normalized"}
    if set(pivot.columns.get_level_values("condition")) != required:
        raise ValueError("both original and normalized conditions are required")

    out = pivot.index.to_frame(index=False)
    for metric in METRICS:
        out[f"{metric}_normalized_minus_original"] = (
            pivot[(metric, "normalized")].to_numpy()
            - pivot[(metric, "original")].to_numpy()
        )
    return out


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Gini values must be nonempty and finite")
    if np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Gini values must be nonnegative with a positive sum")
    values = np.sort(values)
    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    weights = 2 * ranks - values.size - 1
    return float(np.sum(weights * values) / (values.size * values.sum()))


def per_run_inequality(rows: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "seed",
        "normalization_mode",
        "objective",
        "condition",
        "assignment",
    ]
    out = []
    for keys, group in rows.groupby(group_columns, sort=True):
        if group["dataset"].duplicated().any():
            raise ValueError("each run must contain one value per dataset")
        mase = group["final_mase"].to_numpy(dtype=np.float64)
        out.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "n_datasets": int(mase.size),
                "mase_gini": gini(mase),
                "mase_iqr": float(np.quantile(mase, 0.75) - np.quantile(mase, 0.25)),
            }
        )
    return pd.DataFrame(out)


def inequality_condition_effects(rows: pd.DataFrame) -> pd.DataFrame:
    index = ["seed", "normalization_mode", "objective", "assignment"]
    pivot = rows.pivot(index=index, columns="condition")
    required = {"original", "normalized"}
    if set(pivot.columns.get_level_values("condition")) != required:
        raise ValueError("both original and normalized conditions are required")

    out = pivot.index.to_frame(index=False)
    for metric in INEQUALITY_METRICS:
        out[f"{metric}_normalized_minus_original"] = (
            pivot[(metric, "normalized")].to_numpy()
            - pivot[(metric, "original")].to_numpy()
        )
    return out


def seed_summary(
    rows: pd.DataFrame, value: str, group_columns: list[str]
) -> list[dict]:
    summaries = []
    for keys, group in rows.groupby(group_columns, sort=True):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        per_seed = group.groupby("seed")[value].mean().sort_index()
        values = per_seed.to_numpy(dtype=np.float64)
        if values.size < 2:
            raise ValueError("at least two independent seeds are required")
        se = values.std(ddof=1) / np.sqrt(values.size)
        half_width = stats.t.ppf(0.975, values.size - 1) * se
        summaries.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "metric": value,
                "n_seeds": int(values.size),
                "mean": float(values.mean()),
                "ci95_half_width": float(half_width),
                "seed_means": dict(
                    zip(per_seed.index.astype(str), values, strict=True)
                ),
            }
        )
    return summaries


def build_robust_report(jobtag: str, outputs_dir: Path) -> None:
    output_dir = outputs_dir / f"{jobtag}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = read_runs(outputs_dir, jobtag)
    experiment_kinds = set(runs["experiment_kind"])
    if len(experiment_kinds) != 1:
        raise ValueError("one analysis cannot mix experiment kinds")
    experiment_kind = experiment_kinds.pop()
    condition = condition_effects(runs)
    inequality = per_run_inequality(runs)
    inequality_effect = inequality_condition_effects(inequality)
    runs.to_csv(output_dir / "per_run_dataset.csv", index=False)
    condition.to_csv(output_dir / "condition_effects.csv", index=False)
    inequality.to_csv(output_dir / "per_run_inequality.csv", index=False)
    inequality_effect.to_csv(
        output_dir / "inequality_condition_effects.csv", index=False
    )

    summary = {
        "experiment_kind": experiment_kind,
        "scale_effects": [],
        "difference_in_differences": [],
        "condition_effects": [],
        "inequality_condition_effects": [],
    }
    if experiment_kind == "controlled_scale":
        effects = pair_scale_effects(runs)
        did = difference_in_differences(effects)
        effects.to_csv(output_dir / "paired_scale_effects.csv", index=False)
        did.to_csv(output_dir / "difference_in_differences.csv", index=False)
    for metric in METRICS:
        if experiment_kind == "controlled_scale":
            summary["scale_effects"].extend(
                seed_summary(
                    effects,
                    f"{metric}_scale_effect",
                    ["normalization_mode", "objective", "condition"],
                )
            )
            summary["difference_in_differences"].extend(
                seed_summary(
                    did,
                    f"{metric}_did",
                    ["normalization_mode", "objective"],
                )
            )
        summary["condition_effects"].extend(
            seed_summary(
                condition,
                f"{metric}_normalized_minus_original",
                ["normalization_mode", "objective"],
            )
        )
    for metric in INEQUALITY_METRICS:
        summary["inequality_condition_effects"].extend(
            seed_summary(
                inequality_effect,
                f"{metric}_normalized_minus_original",
                ["normalization_mode", "objective"],
            )
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

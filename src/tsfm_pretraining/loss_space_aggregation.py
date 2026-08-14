"""Aggregates GiftEvalPretrain loss-space run outputs into comparison tables.

Reads `summary.json` and `history.json` (per-checkpoint pooled MSE and
dataset/domain/frequency dispersion reports, written by train.py) from a set
of run output directories and produces:

- comparison.csv / comparison.json: one row per run with final pooled MSE,
  log-MSE AUC through step 2000, and final-step Gini + n_sources per
  breakdown level, per the plan's "Dispersion and equity metrics" section.
- paired_effect.json (only with a scale pair): the per-dataset paired
  AUC(assignment A) - AUC(assignment B) effect and its 95% Student-t
  interval across datasets, matching the convention already used for the
  real-data scale-swap crossover in scripts/reproducibility/real_scale_swap
  (see notes/04-real-world-experiment-plan.md).

This module only produces numbers, not figures, following this project's
existing separation between aggregate scripts (numbers) and replot scripts
(figures) -- see
src/plotting/scripts/reproducibility/synthetic_loss_space/replot_metrics.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.tsfm_pretraining import losses as L


def load_run(path: Path) -> tuple[dict, dict]:
    summary = json.loads((path / "summary.json").read_text())
    history = json.loads((path / "history.json").read_text())
    return summary, history


def comparison_row(label: str, summary: dict, history: dict) -> dict:
    steps = np.array(history["step"])
    row = {
        "label": label,
        "final_pooled_mse": summary.get("final_pooled_mse"),
        "log_mse_auc_through_2000": summary.get("log_mse_auc_through_2000"),
        "n_checkpoints": int(steps.size),
    }
    last_report = history["reports"][-1] if history["reports"] else {}
    for level in ("dataset", "domain", "frequency"):
        if level in last_report:
            row[f"final_{level}_gini"] = last_report[level]["gini"]
            row[f"final_{level}_unweighted_mean"] = last_report[level][
                "unweighted_mean"
            ]
            row[f"final_{level}_n_sources"] = last_report[level]["n_sources"]
    return row


def per_dataset_curves(history: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Reconstructs each dataset's (steps, mse) curve from the per-checkpoint
    dispersion reports. A natural-mixture eval sample can miss a small
    dataset at any given checkpoint by chance (see train.py's
    dispersion_report n_sources_min filter), so each dataset's curve is built
    from whichever checkpoints happened to include it, not just checkpoints
    where every dataset was present."""
    per_dataset_steps: dict[str, list[int]] = {}
    per_dataset_values: dict[str, list[float]] = {}
    for step, report in zip(history["step"], history["reports"]):
        for name, value in (
            report.get("dataset", {}).get("per_source_mean_error", {}).items()
        ):
            per_dataset_steps.setdefault(name, []).append(step)
            per_dataset_values.setdefault(name, []).append(value)
    return {
        name: (np.array(per_dataset_steps[name]), np.array(per_dataset_values[name]))
        for name in per_dataset_steps
    }


def paired_auc_effect(history_a: dict, history_b: dict) -> dict:
    curves_a = per_dataset_curves(history_a)
    curves_b = per_dataset_curves(history_b)
    shared = sorted(set(curves_a) & set(curves_b))
    if not shared:
        raise ValueError(
            "no dataset has per-checkpoint dispersion reports in both runs; "
            "increase train.eval_batches so each eval sample covers every dataset"
        )
    effects = {}
    skipped = []
    for name in shared:
        steps_a, mse_a = curves_a[name]
        steps_b, mse_b = curves_b[name]
        cutoff = min(L.TROUGH_STEP_CUTOFF, int(steps_a.max()), int(steps_b.max()))
        if (steps_a <= cutoff).sum() < 2 or (steps_b <= cutoff).sum() < 2:
            skipped.append(name)  # too few in-window checkpoints to integrate an AUC
            continue
        auc_a = L.log_mse_auc(steps_a, mse_a, cutoff_step=cutoff)
        auc_b = L.log_mse_auc(steps_b, mse_b, cutoff_step=cutoff)
        effects[name] = auc_a - auc_b
    if skipped:
        print(
            f"skipped {len(skipped)} dataset(s) with <2 in-window checkpoints: {skipped}"
        )

    values = np.array(list(effects.values()))
    n = values.size
    mean = float(values.mean())
    if n > 1:
        se = float(values.std(ddof=1) / np.sqrt(n))
        ci = float(stats.t.ppf(0.975, df=n - 1) * se)
    else:
        se, ci = float("nan"), float("nan")
    return {
        "per_dataset_auc_effect": effects,
        "n_datasets": n,
        "mean_effect": mean,
        "se": se,
        "ci95": ci,
    }


def aggregate_runs(
    run_entries: list[str], output_dir: Path, scale_pair: str | None
) -> None:
    """Loads every `label=path` entry in run_entries, writes comparison.csv/json
    to output_dir, and (if scale_pair is a `labelA=labelB` entry already
    present in run_entries) writes the paired AUC effect to
    paired_effect.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for entry in run_entries:
        label, path = entry.split("=", 1)
        runs[label] = load_run(Path(path))

    rows = [
        comparison_row(label, summary, history)
        for label, (summary, history) in runs.items()
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    (output_dir / "comparison.json").write_text(
        comparison.to_json(orient="records", indent=2)
    )
    print(comparison.to_string(index=False))

    if scale_pair is not None:
        label_a, label_b = scale_pair.split("=", 1)
        if label_a not in runs or label_b not in runs:
            raise ValueError(
                f"--scale-pair labels must both appear in --run: {label_a}, {label_b}"
            )
        _, history_a = runs[label_a]
        _, history_b = runs[label_b]
        effect = paired_auc_effect(history_a, history_b)
        (output_dir / "paired_effect.json").write_text(json.dumps(effect, indent=2))
        print(
            f"paired AUC effect ({label_a} - {label_b}): "
            f"{effect['mean_effect']:.4f} +/- {effect['ci95']:.4f} (n={effect['n_datasets']} datasets)"
        )

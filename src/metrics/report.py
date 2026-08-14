"""Builds the final scale-free result tables from the recomputed metrics.

Consumes the outputs of src.metrics.scale_free and emits:

- summary.csv / a printed table: final-step pooled nMSE, pooled MASE, and the
  per-dataset Gini on each metric, one row per condition.
- paired_effects.json: for each (normalized, original) x (A, B) pair, the
  per-dataset paired difference and its 95% Student-t interval, matching the
  convention used for the synthetic scale-swap crossover in results.tex.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Each entry is (pair label, condition trained under scale A, under scale B).
SCALE_PAIRS = [
    ("moment_normalized", "moment_normalized_A", "moment_normalized_B"),
    ("moment_original", "moment_original_A", "moment_original_B"),
    ("timesfm_normalized", "timesfm_normalized_A", "timesfm_normalized_B"),
    ("timesfm_original", "timesfm_original_A", "timesfm_original_B"),
]


def paired_effect(values_a: dict, values_b: dict) -> dict:
    """Per-dataset paired difference A - B and its 95% t interval, over the
    datasets where both variants produced a finite value."""
    shared = sorted(
        name
        for name in set(values_a) & set(values_b)
        if np.isfinite(values_a[name]) and np.isfinite(values_b[name])
    )
    diffs = np.array([values_a[name] - values_b[name] for name in shared])
    n = diffs.size
    mean = float(diffs.mean())
    if n > 1:
        se = float(diffs.std(ddof=1) / np.sqrt(n))
        ci = float(stats.t.ppf(0.975, df=n - 1) * se)
        t_stat, p_value = stats.ttest_rel(
            [values_a[name] for name in shared],
            [values_b[name] for name in shared],
        )
    else:
        se = ci = t_stat = p_value = float("nan")
    return {
        "n_datasets": n,
        "mean_effect": mean,
        "se": se,
        "ci95": ci,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "favors_b_count": int((diffs > 0).sum()),
    }


def head_to_head(values_norm: dict, values_orig: dict) -> dict:
    """Rank-based normalized-vs-original comparison on per-dataset error.

    Deliberately rank/ratio based rather than a difference of means. Because
    both variants are scored on identical windows, a per-dataset MASE ratio has an
    identical seasonal-naive denominator top and bottom, so the ratio reduces
    exactly to the raw original-space MAE ratio -- the baseline choice cannot
    bias this comparison, and no unit-ful quantity is ever averaged across
    datasets with different units.
    """
    shared = sorted(
        d
        for d in set(values_norm) & set(values_orig)
        if np.isfinite(values_norm[d])
        and np.isfinite(values_orig[d])
        and values_norm[d] > 0
        and values_orig[d] > 0
    )
    a = np.array([values_norm[d] for d in shared])
    b = np.array([values_orig[d] for d in shared])
    log_ratio = np.log10(a / b)
    wins = int((a < b).sum())
    if a.size > 1:
        w = stats.wilcoxon(a, b)
        p, stat = float(w.pvalue), float(w.statistic)
    else:
        p, stat = float("nan"), float("nan")
    return {
        "n_datasets": int(a.size),
        "normalized_wins": wins,
        "win_rate": wins / a.size if a.size else float("nan"),
        "wilcoxon_stat": stat,
        "wilcoxon_p": p,
        "median_log10_ratio": float(np.median(log_ratio)),
        "median_ratio": float(10 ** np.median(log_ratio)),
        "iqr_log10_ratio": [
            float(np.percentile(log_ratio, 25)),
            float(np.percentile(log_ratio, 75)),
        ],
    }


def build_report(metrics_dir: Path) -> None:
    table = pd.read_csv(metrics_dir / "scale_free_metrics.csv")
    per_dataset = json.loads((metrics_dir / "final_per_dataset.json").read_text())

    final = (
        table.sort_values("step")
        .groupby("label")
        .tail(1)
        .set_index("label")
        .loc[
            :,
            [
                "step",
                "pooled_nmse",
                "pooled_mase",
                "nmse_dataset_gini",
                "nmse_median_dataset_gini",
                "mase_dataset_gini",
                "nmse_dataset_mean",
                "mase_dataset_mean",
                "mase_n_sources",
            ],
        ]
    )
    final.to_csv(metrics_dir / "summary.csv")
    print("=== Final-step scale-free metrics ===")
    print(final.to_string())

    effects = {}
    for pair_label, label_a, label_b in SCALE_PAIRS:
        if label_a not in per_dataset or label_b not in per_dataset:
            continue
        effects[pair_label] = {
            metric: paired_effect(
                per_dataset[label_a][metric], per_dataset[label_b][metric]
            )
            for metric in ("nmse", "nmse_median", "mase")
        }
    (metrics_dir / "paired_effects.json").write_text(json.dumps(effects, indent=2))

    h2h = {}
    for model in ("moment", "timesfm"):
        for setting in ("natural", "A", "B"):
            n_label = f"{model}_normalized_{setting}"
            o_label = f"{model}_original_{setting}"
            if n_label not in per_dataset or o_label not in per_dataset:
                continue
            h2h[f"{model}_{setting}"] = {
                metric: head_to_head(
                    per_dataset[n_label][metric], per_dataset[o_label][metric]
                )
                for metric in ("nmse", "nmse_median", "mase")
            }
    (metrics_dir / "head_to_head.json").write_text(json.dumps(h2h, indent=2))

    print("\n=== Head-to-head: normalized vs original, per dataset (final step) ===")
    print("(ratio < 1 favours normalized space; MASE ratio == raw MAE ratio)")
    for key, by_metric in h2h.items():
        for metric, r in by_metric.items():
            print(
                f"{key:18s} {metric:12s} "
                f"normalized wins {r['normalized_wins']:>2}/{r['n_datasets']:<2} "
                f"median ratio {r['median_ratio']:.3f}x  "
                f"wilcoxon p={r['wilcoxon_p']:.2e}"
            )

    print("\n=== Paired per-dataset effect (scale A - scale B), final step ===")
    for pair_label, by_metric in effects.items():
        for metric, eff in by_metric.items():
            print(
                f"{pair_label:22s} {metric:5s} "
                f"mean={eff['mean_effect']:+.4f} +/- {eff['ci95']:.4f}  "
                f"p={eff['p_value']:.4f}  n={eff['n_datasets']}  "
                f"favors_B={eff['favors_b_count']}"
            )

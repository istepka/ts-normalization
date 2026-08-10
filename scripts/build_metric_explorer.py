"""Renders the interactive metric explorer from a recomputed metrics directory.

Reads the outputs of scripts/recompute_tsfm_scale_free_metrics.py and inlines
them into the explorer template, producing a self-contained HTML page (the
Artifact CSP blocks every external request, so the data has to travel inside
the page).

Usage:
  uv run python -m scripts.build_metric_explorer \
      --metrics-dir outputs/tsfm_scale_free_metrics_v2 \
      --template <template>.html --out <page>.html
"""

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = [
    "pooled_nmse",
    "pooled_mase",
    "nmse_dataset_gini",
    "nmse_median_dataset_gini",
    "mase_dataset_gini",
    "nmse_dataset_mean",
    "nmse_median_dataset_mean",
    "mase_dataset_mean",
    "nmse_domain_gini",
    "mase_domain_gini",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.metrics_dir / "scale_free_metrics.csv")
    per_dataset = json.loads((args.metrics_dir / "final_per_dataset.json").read_text())

    series = {}
    for label, group in table.groupby("label"):
        group = group.sort_values("step")
        series[label] = {"step": [int(s) for s in group["step"]]}
        for metric in METRICS:
            series[label][metric] = [
                None if pd.isna(v) else float(v) for v in group[metric]
            ]

    payload = {"metrics": METRICS, "series": series, "per_dataset": per_dataset}
    template = args.template.read_text()
    if "__DATA__" not in template:
        raise ValueError(f"{args.template} has no __DATA__ placeholder")
    args.out.write_text(
        template.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    )
    print(
        f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB, "
        f"{len(series)} variants, "
        f"{len(next(iter(series.values()))['step'])} checkpoints)"
    )


if __name__ == "__main__":
    main()

"""Renders the interactive metric explorer from a recomputed metrics directory.

Reads the outputs of src.metrics.scale_free and inlines them
into the explorer template, producing a self-contained HTML page (the
Artifact CSP blocks every external request, so the data has to travel inside
the page).
"""

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


def build_explorer(metrics_dir: Path, template: Path, out: Path) -> None:
    table = pd.read_csv(metrics_dir / "scale_free_metrics.csv")
    per_dataset = json.loads((metrics_dir / "final_per_dataset.json").read_text())

    series = {}
    for label, group in table.groupby("label"):
        group = group.sort_values("step")
        series[label] = {"step": [int(s) for s in group["step"]]}
        for metric in METRICS:
            series[label][metric] = [
                None if pd.isna(v) else float(v) for v in group[metric]
            ]

    payload = {"metrics": METRICS, "series": series, "per_dataset": per_dataset}
    template_text = template.read_text()
    if "__DATA__" not in template_text:
        raise ValueError(f"{template} has no __DATA__ placeholder")
    out.write_text(
        template_text.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    )
    print(
        f"wrote {out} ({out.stat().st_size / 1024:.1f} KB, "
        f"{len(series)} variants, "
        f"{len(next(iter(series.values()))['step'])} checkpoints)"
    )

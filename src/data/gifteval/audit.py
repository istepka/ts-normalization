"""Phase 1 of notes/05-timesfm-pretraining-loss-space-plan.md: audit the local
GiftEvalPretrain corpus and fail loudly if it cannot be fingerprinted or the
domain mapping is incomplete.

Produces, under output_dir:
- source_inventory.csv: one row per dataset directory (domain, univariate/
  multivariate, channel count, series count, checksum)
- frequency_inventory.csv: series counts by (dataset, frequency)
- missing_value_stats.csv: per-dataset missing-value fraction summary
- variance_histogram.json: log10-variance histogram over sampled series
- sampling_table.csv: per-dataset/domain/frequency series counts, the basis
  for configuring train.py's dataset_weights
- summary.json: univariate series count and other headline audit numbers
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.gifteval import corpus as gc


def audit_corpus(
    corpus_root: Path,
    output_dir: Path,
    max_series_per_dataset: int | None,
    datasets: str | None,
) -> None:
    domain_map = gc.load_domain_map()
    dataset_names = gc.discover_dataset_dirs(corpus_root)
    print(f"found {len(dataset_names)} dataset directories under {corpus_root}")
    if datasets is not None:
        requested = datasets.split(",")
        missing = set(requested) - set(dataset_names)
        if missing:
            raise ValueError(
                f"--datasets names not found under corpus root: {sorted(missing)}"
            )
        dataset_names = requested

    dataset_infos = [
        gc.describe_dataset(corpus_root, name, domain_map) for name in dataset_names
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    source_inventory = pd.DataFrame(
        [
            {
                "dataset": d.name,
                "domain": d.domain,
                "domain_confidence": d.domain_confidence,
                "is_univariate": d.is_univariate,
                "num_channels": d.num_channels,
                "num_series": d.num_series,
                "checksum": d.checksum,
            }
            for d in dataset_infos
        ]
    )
    source_inventory.to_csv(output_dir / "source_inventory.csv", index=False)

    univariate_infos = [d for d in dataset_infos if d.is_univariate]
    multivariate_infos = [d for d in dataset_infos if not d.is_univariate]
    if not univariate_infos:
        raise RuntimeError(
            "no univariate datasets found in the local GiftEvalPretrain copy; "
            "the corpus cannot be fingerprinted as intended by the plan"
        )
    print(
        f"{len(univariate_infos)} univariate datasets, "
        f"{len(multivariate_infos)} multivariate datasets (excluded from the primary comparison)"
    )

    freq_rows: list[dict] = []
    missing_rows: list[dict] = []
    variances: list[float] = []
    total_series = 0

    for info in univariate_infos:
        freq_counts: dict[str, int] = {}
        missing_fractions: list[float] = []
        n_scanned = 0
        for record in gc.iter_series_records(corpus_root, info.name, info):
            freq_counts[record.frequency] = freq_counts.get(record.frequency, 0) + 1
            missing_fractions.append(record.missing_fraction)
            if np.isfinite(record.variance) and record.variance > 0:
                variances.append(record.variance)
            n_scanned += 1
            if (
                max_series_per_dataset is not None
                and n_scanned >= max_series_per_dataset
            ):
                break

        total_series += n_scanned
        for freq, count in freq_counts.items():
            freq_rows.append(
                {
                    "dataset": info.name,
                    "domain": info.domain,
                    "frequency": freq,
                    "n_series": count,
                }
            )
        missing_rows.append(
            {
                "dataset": info.name,
                "domain": info.domain,
                "n_series_scanned": n_scanned,
                "mean_missing_fraction": float(np.mean(missing_fractions))
                if missing_fractions
                else 0.0,
                "fraction_series_with_any_missing": float(
                    np.mean([f > 0 for f in missing_fractions])
                )
                if missing_fractions
                else 0.0,
            }
        )
        print(
            f"  {info.name}: scanned {n_scanned} series, {len(freq_counts)} frequencies"
        )

    frequency_inventory = pd.DataFrame(freq_rows)
    frequency_inventory.to_csv(output_dir / "frequency_inventory.csv", index=False)

    missing_value_stats = pd.DataFrame(missing_rows)
    missing_value_stats.to_csv(output_dir / "missing_value_stats.csv", index=False)

    log_variances = np.log10(np.asarray(variances))
    hist_counts, hist_edges = np.histogram(log_variances, bins=30)
    (output_dir / "variance_histogram.json").write_text(
        json.dumps(
            {
                "log10_variance_bin_edges": hist_edges.tolist(),
                "counts": hist_counts.tolist(),
                "n_series_sampled": int(log_variances.size),
            },
            indent=2,
        )
    )

    sampling_table = (
        frequency_inventory.groupby(["dataset", "domain"])["n_series"]
        .sum()
        .reset_index()
        .rename(columns={"n_series": "n_series_scanned"})
    )
    sampling_table["configured_dataset_weight"] = 1.0 / len(sampling_table)
    sampling_table.to_csv(output_dir / "sampling_table.csv", index=False)

    domain_counts = (
        source_inventory[source_inventory["is_univariate"]]
        .groupby("domain")["dataset"]
        .count()
    )
    summary = {
        "n_dataset_dirs": len(dataset_names),
        "n_univariate_datasets": len(univariate_infos),
        "n_multivariate_datasets": len(multivariate_infos),
        "n_series_scanned": total_series,
        "max_series_per_dataset_cap": max_series_per_dataset,
        "univariate_datasets_per_domain": domain_counts.to_dict(),
        "domain_map_low_confidence_datasets": sorted(
            name for name, entry in domain_map.items() if entry["confidence"] == "low"
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

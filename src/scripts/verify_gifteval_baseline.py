"""Checks the harness against GIFT-Eval's published seasonal-naive results.

The harness reimplements GIFT-Eval's short-term split rather than importing
their evaluator, which would pull in gluonts. This is what makes that
reimplementation checkable: scoring the same seasonal-naive baseline through
our own loaders, metrics, and seasonality must reproduce the numbers in the
benchmark's `results/seasonal_naive/all_results.csv`.

Any drift here means the split, the seasonality, or the MASE denominator has
moved away from the benchmark, and every GIFT-Eval number the harness
reports is suspect.

Usage:
  uv run python -m src.scripts.verify_gifteval_baseline
"""

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval import predict, score, suites
from src.eval.predict import Forecasts
from src.metrics import accuracy

DEFAULT_ROOT = "/zfsauton/scratch/istepka/lts/data/gift-eval"
# The published results rename some configs: lowercase throughout, no
# `_with_missing` suffix, and saugeenday shortened.
PUBLISHED_RENAMES = {"saugeenday": "saugeen"}


def published_key(config: str) -> str:
    name = config.split("/")[0]
    name = name.removesuffix("_with_missing").lower()
    return PUBLISHED_RENAMES.get(name, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Relative gap above which a config is reported as drifted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    published = pd.read_csv(root / "results/seasonal_naive/all_results.csv")
    published = published[published.dataset.str.endswith("/short")]
    reference = {}
    for dataset, row in zip(published.dataset, published.to_dict("records")):
        name, freq, _ = dataset.split("/")
        reference[(name, freq)] = row

    series = suites.load_gifteval_short(root)
    by_config = collections.defaultdict(list)
    for item in series:
        by_config[item.subset].append(item)

    rows = []
    for config, items in by_config.items():
        horizon = len(items[0].actual)
        actual = np.stack([item.actual for item in items])
        context, valid = predict.build_context(items, args.context_length)
        forecasts = Forecasts(
            values=score.seasonal_naive(items, horizon, 1),
            actual=np.nan_to_num(actual, nan=0.0),
            actual_mask=np.isfinite(actual).astype(float),
            history=context,
            history_mask=valid,
            periods=np.array([item.period for item in items]),
            quantiles=[0.5],
            subsets=[item.subset for item in items],
            item_ids=[item.item_id for item in items],
        )
        pooled = accuracy.pool(score.score(forecasts, items))

        name = published_key(config)
        freq = config.split("/")[1] if "/" in config else None
        if freq is None:
            candidates = [k for k in reference if k[0] == name]
            if len(candidates) != 1:
                raise ValueError(f"{config}: {len(candidates)} published matches")
            key = candidates[0]
        else:
            key = (name, freq)
        entry = reference[key]
        # WQL is compared against their ND column, not their WQL column. Our
        # baseline is a median-only forecast, and for a single 0.5 quantile
        # the doubled pinball loss reduces to the absolute error, so WQL is
        # ND by definition. Their WQL is over a 9-quantile distribution and
        # is not the same quantity.
        rows.append(
            {
                "config": config,
                "mase": pooled["mase"] / entry["eval_metrics/MASE[0.5]"],
                "mae": pooled["mae"] / entry["eval_metrics/MAE[0.5]"],
                "wql": pooled["wql"] / entry["eval_metrics/ND[0.5]"],
            }
        )

    table = pd.DataFrame(rows).sort_values("config").replace([np.inf, -np.inf], np.nan)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nratio ours/published:")
    drifted = {}
    for metric in ("mase", "mae", "wql"):
        values = table[metric].dropna()
        gap = (values - 1).abs()
        drifted[metric] = int((gap >= args.tolerance).sum())
        print(
            f"  {metric:5s} median {values.median():.4f}  within 1%: "
            f"{(gap < 0.01).sum():2d}/{len(values)}  within "
            f"{args.tolerance:.0%}: {(gap < args.tolerance).sum():2d}/{len(values)}"
        )
    if any(drifted.values()):
        print(f"\nconfigs beyond tolerance: {drifted}")


if __name__ == "__main__":
    main()

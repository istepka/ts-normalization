"""Scores the classical reference baselines over the held-out suites.

Writes the same `eval_by_subset.csv` schema `run_tsfm_eval.py` writes, one
directory per baseline, so the collector reads model and baseline rows
through the same path and every number comes from the same metric code.

Only `native` mode. The `fixed` and `rolling` modes exist to cross-check a
model against its training window and to measure forecast stability under
re-forecasting, neither of which says anything about an estimator refitted
per series.

Usage:
  uv run python -m src.scripts.run_baseline_eval --baseline arima \
      --output-dir <dir> [--suite m1 --suite m3]
"""

import argparse
import collections
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval import baselines, predict, score, suites
from src.metrics import accuracy

ROOTS = {
    "monash": "/zfsauton/scratch/istepka/lts/data/monash_eval",
    "m4": "/zfsauton/scratch/istepka/lts/data/m4_official",
    "gifteval": "/zfsauton/scratch/istepka/lts/data/gift-eval",
    "corpus": "/zfsauton/scratch/istepka/lts/data/giftevalpretrain_full",
}
# Only used to shape the history array the scorer reads for nMSE. The
# baselines themselves see each series' full history, not this window.
CONTEXT_LENGTH = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=baselines.BASELINES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", action="append", dest="suites", default=None)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _one(payload):
    name, item, horizon = payload
    return baselines.forecast_series(name, item, horizon)


def score_subset(name: str, items: list, workers: int) -> dict:
    horizon = len(items[0].actual)
    payloads = [(name, item, horizon) for item in items]
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            values = pool.map(_one, payloads, chunksize=32)
    else:
        values = [_one(p) for p in payloads]
    values = np.stack(values)

    actual = np.stack([item.actual for item in items])
    context, valid = predict.build_context(items, CONTEXT_LENGTH)
    forecasts = predict.Forecasts(
        values=values,
        actual=np.nan_to_num(actual, nan=0.0),
        actual_mask=np.isfinite(actual).astype(float),
        history=context,
        history_mask=valid,
        periods=np.array([item.period for item in items]),
        quantiles=baselines.QUANTILES,
        subsets=[item.subset for item in items],
        item_ids=[item.item_id for item in items],
    )
    pooled = accuracy.pool(score.score(forecasts, items))
    # A fixed-form estimator does not fit every real series. Reporting the
    # share that produced a forecast keeps a baseline column honest.
    fitted = int(np.isfinite(values[:, 0, 0]).sum())
    return {"n_series": len(items), "n_fitted": fitted, **pooled}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chosen = args.suites if args.suites else list(suites.SUITES)

    records = []
    for suite in chosen:
        series = suites.load_suite(suite, ROOTS)
        grouped = collections.defaultdict(list)
        for item in series:
            grouped[item.subset].append(item)
        print(f"{suite}: {len(series)} series, {len(grouped)} subsets", flush=True)
        for subset, items in sorted(grouped.items()):
            row = score_subset(args.baseline, items, args.workers)
            freq = sorted({i.freq for i in items if i.freq is not None})
            records.append(
                {
                    "suite": suite,
                    "mode": "native",
                    "subset": subset,
                    "freq": freq[0] if freq else "none",
                    **row,
                }
            )
            print(
                f"  {subset}: mase {row.get('mase', float('nan')):.4f} "
                f"({row['n_fitted']}/{row['n_series']} fitted)",
                flush=True,
            )

    table = pd.DataFrame(records)
    table.insert(0, "seed", 0)
    table.insert(0, "condition", args.baseline)
    table.insert(0, "model", args.baseline)
    table.to_csv(args.output_dir / "eval_by_subset.csv", index=False)
    (args.output_dir / "eval_summary.json").write_text(
        json.dumps({"baseline": args.baseline, "suites": chosen}, indent=2)
    )
    print(f"\nwrote {args.output_dir / 'eval_by_subset.csv'}")


if __name__ == "__main__":
    main()

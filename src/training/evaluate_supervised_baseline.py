"""Evaluate one standalone statistical reference over rolling test origins."""

import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np

from src.eval import baselines, predict, score
from src.eval.suites import EvalSeries
from src.metrics import accuracy
from src.supervised.data import (
    context_length,
    eligible_series,
    frequency_groups,
    load_series,
    model_horizon,
    split_series,
)


def _forecast(payload: tuple[str, EvalSeries, int]) -> np.ndarray:
    return baselines.forecast_series(*payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=("ets", "arima"), required=True)
    parser.add_argument("--suite", choices=("m1", "m3", "m4", "tourism"), required=True)
    parser.add_argument(
        "--frequency", choices=("Y", "Q", "M", "W", "D", "H"), required=True
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--monash-root",
        type=Path,
        default=Path("/zfsauton/scratch/istepka/lts/data/monash_eval"),
    )
    parser.add_argument(
        "--m4-root",
        type=Path,
        default=Path("/zfsauton/scratch/istepka/lts/data/m4_official"),
    )
    args = parser.parse_args()

    series = load_series(
        args.monash_root,
        (args.suite,),
        args.m4_root if args.suite == "m4" else None,
    )
    source = frequency_groups(series)[args.frequency]
    horizon = model_horizon(source)
    input_size = context_length(source)
    eligible = eligible_series(source, horizon, input_size + horizon)
    splits = split_series(eligible, horizon)[args.shard_index :: args.num_shards]
    if not splits:
        raise ValueError("baseline shard contains no series")

    rows = []
    with multiprocessing.Pool(args.workers) as pool:
        for origin_offset in range(max(s.item.official_horizon for s in splits)):
            active = [
                split for split in splits if origin_offset < split.item.official_horizon
            ]
            horizons = sorted({split.item.official_horizon for split in active})
            for official_horizon in horizons:
                items = []
                for split in active:
                    if split.item.official_horizon != official_horizon:
                        continue
                    start = split.test_start + origin_offset
                    items.append(
                        EvalSeries(
                            suite=split.item.suite,
                            subset=split.item.subset,
                            item_id=split.item.item_id,
                            history=split.item.values[:start],
                            actual=split.item.values[start : start + official_horizon],
                            period=split.item.period,
                            freq=split.item.freq,
                        )
                    )
                values = np.stack(
                    pool.map(
                        _forecast,
                        [(args.baseline, item, official_horizon) for item in items],
                        chunksize=16,
                    )
                )
                median_index = baselines.QUANTILES.index(0.5)
                values = values[:, :, median_index : median_index + 1]
                context, context_mask = predict.build_context(items, input_size)
                forecasts = predict.Forecasts(
                    values=values,
                    actual=np.stack([item.actual for item in items]),
                    actual_mask=np.ones((len(items), official_horizon)),
                    history=context,
                    history_mask=context_mask,
                    periods=np.array([item.period for item in items]),
                    quantiles=[0.5],
                    subsets=[item.subset for item in items],
                    item_ids=[item.item_id for item in items],
                )
                scores = score.score(forecasts, items)
                pooled = accuracy.pool(scores)
                usable = (
                    np.isfinite(scores["wql_num"])
                    & np.isfinite(scores["wql_den"])
                    & (scores["wql_den"] > 0)
                )
                rows.append(
                    {
                        "origin_offset": origin_offset,
                        "horizon": official_horizon,
                        "n_series": len(items),
                        "n_fitted": int(np.isfinite(values[:, 0, 0]).sum()),
                        **pooled,
                        "wql_num": float(scores["wql_num"][usable].sum()),
                        "wql_den": float(scores["wql_den"][usable].sum()),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (
        f"{args.baseline}_{args.suite}_{args.frequency}_"
        f"{args.shard_index:03d}-of-{args.num_shards:03d}.json"
    )
    output.write_text(
        json.dumps(
            {
                "baseline": args.baseline,
                "suite": args.suite,
                "frequency": args.frequency,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "rows": rows,
            },
            indent=2,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()

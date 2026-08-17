"""CLI for src.data.gifteval.window_index.build_and_save_window_index:
pre-builds and caches the canonical GiftEvalPretrain window index.

Usage:
  uv run python -m src.scripts.build_gifteval_window_index \
      --corpus-root /zfsauton/scratch/istepka/lts/data/giftevalpretrain_full \
      --output outputs/gifteval_window_index/context512_pred128.parquet \
      --context-length 512 --prediction-length 128 --stride 512
"""

import argparse
from pathlib import Path

from src.data.gifteval import window_index as wi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated; omit for every univariate dataset.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated datasets held out for evaluation; must match "
        "conf/tsfm_base.yaml's corpus.exclude.",
    )
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--val-series-fraction", type=float, default=0.1)
    parser.add_argument("--min-valid-fraction", type=float, default=0.9)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--max-windows-per-series", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = wi.WindowIndexConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        val_series_fraction=args.val_series_fraction,
        min_valid_fraction=args.min_valid_fraction,
        base_seed=args.base_seed,
        max_windows_per_series=args.max_windows_per_series,
    )
    wi.build_and_save_window_index(
        args.corpus_root, args.output, args.datasets, config, args.exclude
    )


if __name__ == "__main__":
    main()

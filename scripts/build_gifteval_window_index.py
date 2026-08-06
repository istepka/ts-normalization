"""Pre-builds and caches the canonical GiftEvalPretrain window index.

Paired runs (moment_normalized vs moment_original, MOMENT vs TimesFM, scale
assignment A vs B) must all train on the same base index (see the plan's
"Canonical window index" section and Data test "Paired conditions consume
identical base windows"). Building it once here and passing the resulting
parquet path via train.py's `window_index.cache_path` avoids every paired job
independently re-scanning the corpus and risking a mismatch.

Usage:
  uv run python -m scripts.build_gifteval_window_index \
      --corpus-root /zfsauton/scratch/istepka/lts/data/giftevalpretrain_full \
      --output outputs/gifteval_window_index/context512_pred128.parquet \
      --context-length 512 --prediction-length 128 --stride 512
"""

import argparse
from pathlib import Path

from src.tsfm_pretraining import gifteval_corpus as gc
from src.tsfm_pretraining import window_index as wi


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
    domain_map = gc.load_domain_map()
    dataset_names = (
        args.datasets.split(",")
        if args.datasets is not None
        else gc.discover_dataset_dirs(args.corpus_root)
    )
    config = wi.WindowIndexConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        val_series_fraction=args.val_series_fraction,
        min_valid_fraction=args.min_valid_fraction,
        base_seed=args.base_seed,
        max_windows_per_series=args.max_windows_per_series,
    )
    print(f"building window index over {len(dataset_names)} datasets ...")
    print(
        f"window_length = context_length + prediction_length = "
        f"{args.context_length + args.prediction_length} raw points; "
        "any series shorter than that contributes zero windows"
    )
    index = wi.build_window_index(args.corpus_root, dataset_names, domain_map, config)
    index.save(args.output)

    n_train, n_val = len(index.split("train")), len(index.split("val"))
    print(f"n_windows={len(index)} n_train={n_train} n_val={n_val}")

    covered = set(index.table["dataset"].unique())
    zero_window_datasets = sorted(set(dataset_names) - covered)
    if zero_window_datasets:
        print(
            f"WARNING: {len(zero_window_datasets)} requested dataset(s) contributed "
            f"zero windows (series too short for window_length, or all-multivariate): "
            f"{zero_window_datasets}"
        )
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()

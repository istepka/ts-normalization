"""Loads all six evaluation suites and reports their shape.

The cheap checks live in tests/test_eval_suites.py. This script covers the
expensive ones (M4 at 100k series, Favorita, GIFT-Eval at 55 configs) that
are too slow for the test suite, and is the thing to re-run after any change
to a data path or a loader.

Usage:
  uv run python -m src.scripts.verify_eval_suites
"""

import argparse

import numpy as np

from src.eval import suites

DEFAULT_ROOTS = {
    "monash": "/zfsauton/scratch/istepka/lts/data/monash_eval",
    "m4": "/zfsauton/scratch/istepka/lts/data/m4_official",
    "gifteval": "/zfsauton/scratch/istepka/lts/data/gift-eval",
    "corpus": "/zfsauton/scratch/istepka/lts/data/giftevalpretrain_full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suites",
        type=str,
        default=",".join(suites.SUITES),
        help="Comma-separated subset to check.",
    )
    for key, value in DEFAULT_ROOTS.items():
        parser.add_argument(f"--{key}-root", type=str, default=value)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = {key: getattr(args, f"{key}_root") for key in DEFAULT_ROOTS}

    print(
        f"{'suite':10s} {'series':>9s} {'subsets':>8s} {'horizons':>22s} {'history len'}"
    )
    for name in args.suites.split(","):
        series = suites.load_suite(name, roots)
        lengths = np.array([len(s.history) for s in series])
        horizons = sorted({len(s.actual) for s in series})
        shown = horizons if len(horizons) <= 5 else f"{horizons[0]}..{horizons[-1]}"
        print(
            f"{name:10s} {len(series):9d} {len({s.subset for s in series}):8d} "
            f"{shown!s:>22s} "
            f"min={lengths.min()} med={int(np.median(lengths))} max={lengths.max()}"
        )
        short = int((lengths < 512).sum())
        if short:
            print(
                f"{'':10s} {short} series ({100 * short / len(series):.1f}%) have "
                "under 512 points of history and must be padded, not dropped"
            )


if __name__ == "__main__":
    main()

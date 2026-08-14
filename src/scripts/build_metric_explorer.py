"""CLI for src.metrics.explorer: renders the interactive
metric explorer from a recomputed metrics directory. See that module's
docstring for details.

Usage:
  uv run python -m src.scripts.build_metric_explorer \
      --metrics-dir outputs/YYYY-MM-DD/analysis/tsfm_pretraining/scale_free_metrics \
      --template <template>.html --out <page>.html
"""

import argparse
from pathlib import Path

from src.metrics.explorer import build_explorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_explorer(args.metrics_dir, args.template, args.out)


if __name__ == "__main__":
    main()

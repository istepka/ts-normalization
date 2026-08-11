"""CLI for src.tsfm_pretraining.scale_free_report: builds the final
scale-free result tables from the recomputed metrics. See that module's
docstring for what it produces.

Usage:
  uv run python -m scripts.report_scale_free_tables \
      --metrics-dir outputs/tsfm_scale_free_metrics
"""

import argparse
from pathlib import Path

from src.tsfm_pretraining.scale_free_report import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(args.metrics_dir)


if __name__ == "__main__":
    main()

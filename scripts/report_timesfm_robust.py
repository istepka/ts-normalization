"""CLI for src.tsfm_pretraining.timesfm_robust_report: analyzes replicated
TimesFM controlled-scale runs. See that module's docstring for details.
"""

import argparse
from pathlib import Path

from src.tsfm_pretraining.timesfm_robust_report import build_robust_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobtag", required=True)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_robust_report(args.jobtag, args.outputs_dir)


if __name__ == "__main__":
    main()

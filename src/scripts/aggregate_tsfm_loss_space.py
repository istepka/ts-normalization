"""CLI for src.metrics.aggregate: aggregates
GiftEvalPretrain loss-space run outputs into comparison tables. See that
module's docstring for what it produces.

Usage:
  uv run python -m src.scripts.aggregate_tsfm_loss_space \
      --run moment_normalized=outputs/YYYY-MM-DD/experiments/tsfm_pretraining/.../moment_normalized \
      --run moment_original=outputs/YYYY-MM-DD/experiments/tsfm_pretraining/.../moment_original \
      --output-dir outputs/YYYY-MM-DD/analysis/tsfm_pretraining/.../aggregate

  # Paired controlled-scale effect (also requires --run entries for context):
  uv run python -m src.scripts.aggregate_tsfm_loss_space \
      --run moment_original_A=outputs/YYYY-MM-DD/experiments/tsfm_pretraining/.../moment_original_A \
      --run moment_original_B=outputs/YYYY-MM-DD/experiments/tsfm_pretraining/.../moment_original_B \
      --scale-pair moment_original_A=moment_original_B \
      --output-dir outputs/YYYY-MM-DD/analysis/tsfm_pretraining/.../aggregate_controlled_scale
"""

import argparse
from pathlib import Path

from src.metrics.aggregate import aggregate_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="label=path",
        help="Repeatable. A run's wandb-run label and its output_dir.",
    )
    parser.add_argument(
        "--scale-pair",
        default=None,
        metavar="labelA=labelB",
        help="Two --run labels (already run under scale assignment A and B) to compute "
        "the paired per-dataset AUC effect for.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_runs(args.run, args.output_dir, args.scale_pair)


if __name__ == "__main__":
    main()

"""CLI for src.tsfm_pretraining.scale_free_metrics: recomputes scale-free eval
metrics (nMSE, MASE) from saved checkpoints. See that module's docstring for
why this recompute is needed.

Usage:
  uv run python -m scripts.recompute_tsfm_scale_free_metrics \
      --run moment_original_A=outputs/gifteval_moment_..._A \
      --run moment_original_B=outputs/gifteval_moment_..._B \
      --output-dir outputs/scale_free_metrics
"""

import argparse
from pathlib import Path

from src.tsfm_pretraining.scale_free_metrics import recompute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="label=path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Windows per forward pass; bounds peak activation memory.",
    )
    parser.add_argument(
        "--pooled-windows",
        type=int,
        default=4096,
        help="Cap on the natural-mixture pooled sample. The runs used "
        "eval_batches * batch_size (25,600), whose per-window series loads "
        "dominate runtime; the pooled metric is secondary to the per-dataset "
        "Gini, which comes from the stratified sample and is not capped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recompute_metrics(
        args.run, args.output_dir, args.device, args.chunk_size, args.pooled_windows
    )


if __name__ == "__main__":
    main()

"""CLI for src.tsfm_pretraining.corpus_audit: audits the local GiftEvalPretrain
corpus and fails loudly if it cannot be fingerprinted or the domain mapping
is incomplete. See that module's docstring for what it produces.

Usage:
  uv run python -m src.tsfm_pretraining.scripts.audit_gifteval_pretrain \
      --corpus-root /zfsauton/scratch/istepka/lts/data/giftevalpretrain_full \
      --output-dir outputs/gifteval_audit

For a fast development pass (not the real Phase 1 deliverable), cap series
scanned per dataset with --max-series-per-dataset.
"""

import argparse
from pathlib import Path

from src.tsfm_pretraining.corpus_audit import audit_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-series-per-dataset",
        type=int,
        default=None,
        help="Cap series scanned per dataset (development only; omit for the real audit).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names to audit (development only; omit for the real, full audit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_corpus(
        args.corpus_root,
        args.output_dir,
        args.max_series_per_dataset,
        args.datasets,
    )


if __name__ == "__main__":
    main()

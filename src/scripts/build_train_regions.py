"""Builds the train-region corpus root that pretraining reads.

Emits one Arrow dataset per held-out suite subset holding only each series'
train region (see src/data/gifteval/train_regions.py for the split rule),
then assembles a root that symlinks every canonical corpus directory beside
them. The canonical copies stay in `corpus.exclude`, so the exclusion list
does not move and the emitted regions are the only route those competitions
have into the index.

Every emitted series is audited against the canonical one before the root is
written: it must be a strict prefix, and it must stop before the first point
any evaluation protocol scores.

  uv run python -m src.scripts.build_train_regions \
      --out-root /zfsauton/scratch/istepka/lts/data/giftevalpretrain_trainsplit
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.gifteval import train_regions as tr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("/zfsauton/scratch/istepka/lts/data/giftevalpretrain_full"),
    )
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
    parser.add_argument("--out-root", type=Path, required=True)
    return parser.parse_args()


def audit(regions: list[tr.TrainRegion]) -> dict:
    """Fails on any region that reaches a scored point.

    The evaluated horizon is the final `official_horizon` points, and the
    train region is required to stop `2H - 1 + validation_size` short of the
    end, so the margin between the two is the reserved test and validation
    region. A region with zero margin would still be leakage-free under the
    competition protocol but would break the comparison with the supervised
    split, so the stricter bound is the one asserted.

    Regions marked not evaluated are skipped, not waived: those series are
    absent from the suite the harness loads, so they have no scored point to
    stop before.
    """
    margins = []
    for region in regions:
        if not region.evaluated:
            continue
        limit = tr.train_end_index(
            region.canonical_length, region.official_horizon, region.validation_size
        )
        if len(region.values) > limit:
            raise ValueError(
                f"{region.dataset}/{region.item_id}: train region of "
                f"{len(region.values)} points reaches past its limit of {limit}"
            )
        margins.append(region.canonical_length - len(region.values))
    return {
        "n_series": len(regions),
        "n_evaluated_series": len(margins),
        "min_reserved_points": int(np.min(margins)),
        "median_reserved_points": float(np.median(margins)),
    }


def link_canonical(corpus_root: Path, out_root: Path) -> int:
    """Symlinks every canonical corpus directory into the new root.

    Symlinks rather than copies because the canonical corpus is hundreds of
    gigabytes and none of it changes here. The excluded competition
    directories are linked too: `discover_dataset_dirs` raises on an
    exclusion naming a directory it cannot see.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    linked = 0
    for source in sorted(corpus_root.iterdir()):
        destination = out_root / source.name
        if destination.exists() or destination.is_symlink():
            continue
        destination.symlink_to(source)
        linked += 1
    return linked


def main() -> None:
    args = parse_args()
    regions = tr.monash_and_m4_regions(args.monash_root, args.m4_root)
    regions.extend(tr.favorita_regions(args.corpus_root))
    summary = audit(regions)

    counts = tr.write_regions(regions, args.out_root)
    summary["linked_canonical_dirs"] = link_canonical(args.corpus_root, args.out_root)
    summary["emitted"] = counts
    (args.out_root / "train_regions_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

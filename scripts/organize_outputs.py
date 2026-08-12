"""Move quiescent output directories into the date/category layout.

The command is intentionally dry-run by default. It only considers immediate
children of the outputs root and refuses to overwrite an existing destination.
Use a cutoff date to leave today's potentially active outputs untouched.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

CATEGORIES = {"analysis", "data", "diagnostics", "experiments", "visualizations"}
DATE_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def classify(name: str) -> tuple[str, str]:
    suffix = Path(name).suffix.lower()
    if suffix in {".gif", ".html", ".pdf", ".png", ".svg"}:
        return "visualizations", "loose_artifacts"
    if suffix in {".csv", ".json", ".md", ".npz", ".parquet"}:
        return "analysis", "loose_artifacts"
    if name == "gifteval_window_index":
        return "data", "gifteval_window_index"
    if name.endswith("_paper") or name in {
        "per_dataset_mase_plots",
        "tsfm_natural_convergence",
    }:
        return "visualizations", "paper_or_plots"
    if any(
        marker in name
        for marker in (
            "_aggregate",
            "_analysis",
            "_comparison",
            "_metrics",
            "_replot",
            "permutation_analysis",
        )
    ):
        return "analysis", "derived_reports"
    if name.startswith(("gpu_", "loss_early_sanity")):
        return "diagnostics", "smoke_and_throughput"
    return "experiments", "legacy_runs"


def output_date(path: Path) -> str:
    local_tz = datetime.now().astimezone().tzinfo
    return datetime.fromtimestamp(path.stat().st_mtime, tz=local_tz).strftime(
        "%Y-%m-%d"
    )


def destination(root: Path, source: Path, source_date: str) -> Path:
    category, experiment = classify(source.name)
    return root / source_date / category / experiment / source.name


def plan_moves(root: Path, cutoff: str, excluded: set[str]) -> list[tuple[Path, Path]]:
    moves = []
    for source in sorted(root.iterdir()):
        if source.name in excluded or source.name == "organization_manifest.json":
            continue
        if source.is_symlink():
            continue
        if DATE_NAME.fullmatch(source.name):
            if source.name >= cutoff:
                continue
            legacy_root = source / "legacy" / "hydra"
            for child in sorted(source.iterdir()):
                if child.name == "legacy" or child.name in CATEGORIES:
                    continue
                if not child.is_dir():
                    continue
                target = legacy_root / child.name
                moves.append((child, target))
            continue
        if output_date(source) >= cutoff:
            continue
        moves.append((source, destination(root, source, output_date(source))))
    return moves


def apply_moves(moves: list[tuple[Path, Path]]) -> list[dict[str, str]]:
    for _, target in moves:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to overwrite {target}")

    manifest = []
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        manifest.append({"source": str(source), "target": str(target)})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Immediate output directory name to leave untouched. Repeatable.",
    )
    parser.add_argument(
        "--before",
        required=True,
        help="Only move entries dated before this YYYY-MM-DD cutoff.",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not DATE_NAME.fullmatch(args.before):
        raise ValueError(f"invalid cutoff date: {args.before}")
    root = args.outputs_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    moves = plan_moves(root, args.before, set(args.exclude))
    for source, target in moves:
        print(f"{source} -> {target}")
    print(f"planned_moves={len(moves)} apply={args.apply}")
    if not args.apply:
        return

    manifest = apply_moves(moves)
    manifest_path = root / "organization_manifest.json"
    previous_moves = []
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        previous_moves = previous["moves"]
    manifest_path.write_text(
        json.dumps(
            {
                "cutoff": args.before,
                "moves": previous_moves + manifest,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()

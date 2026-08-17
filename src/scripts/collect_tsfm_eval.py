"""Collects per-run eval tables into one comparison of SIT against RevIN.

`run_tsfm_eval` scores one checkpoint and writes one `eval_by_subset.csv`.
This concatenates those across seeds and conditions and reports the number
the experiment is actually about, at two grains: one row per benchmark for
the main paper, and benchmark by frequency for the appendix.

Seeds are averaged rather than pooled because each seed is an independent
training run, so the spread across them is the error bar on the comparison
and is reported alongside the mean.

Usage:
  uv run python -m src.scripts.collect_tsfm_eval --eval-root <dir>
"""

import argparse
from pathlib import Path

import pandas as pd

# The headline columns. Anything else stays in the concatenated CSV rather
# than the report, which exists to be read rather than to be complete.
# Every metric here is lower-is-better, so the SIT-over-RevIN ratio reads the
# same way throughout: below 1 favors SIT.
ACCURACY_METRICS = ("mase", "wql", "crps", "nmse", "mae", "smape")
STABILITY_METRICS = ("excess_volatility", "sfpc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_runs(eval_root: Path) -> pd.DataFrame:
    tables = sorted(eval_root.glob("*/eval_by_subset.csv"))
    if not tables:
        raise FileNotFoundError(f"no eval_by_subset.csv under {eval_root}")
    frames = []
    for path in tables:
        frame = pd.read_csv(path)
        frame["run"] = path.parent.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def collapse(
    table: pd.DataFrame, metrics: tuple[str, ...], grain: tuple[str, ...]
) -> pd.DataFrame:
    """One row per (run, *grain, mode), weighting subsets by series count.

    `grain` is `("suite",)` for the main table and `("suite", "freq")` for
    the appendix breakdown. Subsets are weighted rather than averaged flat
    because the suites are unbalanced enough that a flat mean over
    GIFT-Eval's 55 configs is a different claim than a mean over its
    instances.
    """
    present = [name for name in metrics if name in table.columns]
    counts = pd.to_numeric(table["n_series"], errors="coerce").fillna(0.0)
    keys = ("model", "condition", "seed", *grain, "mode")
    rows = []
    for values, group in table.groupby(list(keys)):
        weight = counts.loc[group.index]
        entry = dict(zip(keys, values))
        entry["n_series"] = float(weight.sum())
        for name in present:
            column = pd.to_numeric(group[name], errors="coerce")
            usable = column.notna() & (weight > 0)
            if usable.any():
                entry[name] = float(
                    (column[usable] * weight[usable]).sum() / weight[usable].sum()
                )
        rows.append(entry)
    return pd.DataFrame(rows)


def compare_conditions(
    per_run: pd.DataFrame, metrics: tuple[str, ...], grain: tuple[str, ...]
) -> pd.DataFrame:
    """SIT against RevIN per (*grain, mode), averaged over seeds.

    The run's condition names the loss space, and `NormalizationModule.module`
    maps `<model>_normalized` to `SIT` and `<model>_original` to `RevIN`, so
    the two names are the same distinction under different labels. TimesFM's
    original-space condition is `timesfm_native_original`, hence the suffix
    test rather than an equality.
    """
    present = [name for name in metrics if name in per_run.columns]
    per_run = per_run.assign(
        space=per_run["condition"].map(
            lambda name: "sit" if name.endswith("normalized") else "revin"
        )
    )
    keys = ("model", *grain, "mode")
    rows = []
    for values, group in per_run.groupby(list(keys)):
        entry = dict(zip(keys, values))
        entry["n_seeds"] = int(group["seed"].nunique())
        for name in present:
            for space in ("sit", "revin"):
                column = pd.to_numeric(
                    group.loc[group["space"] == space, name], errors="coerce"
                ).dropna()
                entry[f"{name}_{space}"] = (
                    float(column.mean()) if len(column) else float("nan")
                )
                entry[f"{name}_{space}_std"] = (
                    float(column.std(ddof=1)) if len(column) > 1 else float("nan")
                )
            revin = entry[f"{name}_revin"]
            entry[f"{name}_ratio"] = (
                entry[f"{name}_sit"] / revin if revin else float("nan")
            )
        rows.append(entry)
    return pd.DataFrame(rows)


def render_report(
    comparison: pd.DataFrame,
    per_run: pd.DataFrame,
    grain: tuple[str, ...],
    title: str,
) -> str:
    """A markdown table per metric, since a single wide table is unreadable.

    Every table is SIT against RevIN, which is the comparison the experiment
    is about. `grain` decides the row key: benchmark alone for the main
    tables, benchmark and frequency for the appendix ones.
    """
    lines = [f"# {title}", ""]
    runs = per_run[["model", "condition", "seed"]].drop_duplicates()
    lines.append(f"{len(runs)} runs scored, {per_run['seed'].nunique()} seeds.")
    lines.append("")
    lines.append(
        "Every metric is lower-is-better, so a ratio below 1 favors SIT. "
        "Mean over seeds, standard deviation in brackets."
    )
    lines.append("")

    header = " | ".join(("model", *grain, "mode", "SIT", "RevIN", "SIT/RevIN"))
    divider = "|".join(["---"] * (len(grain) + 5))
    for name in ACCURACY_METRICS + STABILITY_METRICS:
        column = f"{name}_ratio"
        if column not in comparison.columns:
            continue
        block = comparison[comparison[column].notna()]
        if block.empty:
            continue
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"| {header} |")
        lines.append(f"|{divider}|")
        for _, row in block.sort_values(["model", *grain, "mode"]).iterrows():
            cells = [row["model"], *(str(row[key]) for key in grain), row["mode"]]
            lines.append(
                "| "
                + " | ".join(cells)
                + f" | {row[f'{name}_sit']:.4f} [{row[f'{name}_sit_std']:.4f}]"
                + f" | {row[f'{name}_revin']:.4f} [{row[f'{name}_revin_std']:.4f}]"
                + f" | {row[column]:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


# The two deliverables: the main paper reports one row per benchmark, the
# appendix breaks the same comparison out by frequency.
GRAINS = {
    "main": (("suite",), "TSFM held-out evaluation by benchmark"),
    "by_frequency": (
        ("suite", "freq"),
        "TSFM held-out evaluation by benchmark and frequency",
    ),
}


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir if args.output_dir is not None else args.eval_root
    out_dir.mkdir(parents=True, exist_ok=True)

    table = load_runs(args.eval_root)
    metrics = ACCURACY_METRICS + STABILITY_METRICS
    table.to_csv(out_dir / "eval_all_runs.csv", index=False)
    print(f"wrote {out_dir / 'eval_all_runs.csv'}")

    for label, (grain, title) in GRAINS.items():
        per_run = collapse(table, metrics, grain)
        comparison = compare_conditions(per_run, metrics, grain)
        per_run.to_csv(out_dir / f"eval_by_run_{label}.csv", index=False)
        comparison.to_csv(out_dir / f"eval_comparison_{label}.csv", index=False)
        report = render_report(comparison, per_run, grain, title)
        (out_dir / f"eval_report_{label}.md").write_text(report)
        print(f"\n{report}")
        print(f"wrote {out_dir / f'eval_comparison_{label}.csv'}")
        print(f"wrote {out_dir / f'eval_report_{label}.md'}")


if __name__ == "__main__":
    main()

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

    Both a weighted mean and an unweighted median across subsets are emitted,
    the median under a `_median` suffix. The mean alone is not a safe summary
    of nMSE: its per-series denominator is the context variance and is
    unbounded, so on the first real run `m4_daily` carried a subset nMSE of
    2.0e7 and dragged the M4 mean to 1.4e7 while the median over 47 subsets
    sat at 1.55, against a training-time 1.50. MASE and WQL have bounded
    denominators and their means are trustworthy.
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
                entry[f"{name}_median"] = float(column[usable].median())
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

    Returns one row per (*grain, mode, space) rather than pairing SIT and
    RevIN into the same row. The paired layout crammed two numbers and a
    ratio into each cell and was awkward to read; a column of one number
    each scans down cleanly.
    """
    present = [
        name
        for base in metrics
        for name in (base, f"{base}_median")
        if name in per_run.columns
    ]
    per_run = per_run.assign(
        space=per_run["condition"].map(
            lambda name: "SIT" if name.endswith("normalized") else "RevIN"
        )
    )
    keys = ("model", *grain, "mode", "space")
    rows = []
    for values, group in per_run.groupby(list(keys)):
        entry = dict(zip(keys, values))
        entry["n_seeds"] = int(group["seed"].nunique())
        for name in present:
            column = pd.to_numeric(group[name], errors="coerce").dropna()
            entry[name] = float(column.mean()) if len(column) else float("nan")
            entry[f"{name}_std"] = (
                float(column.std(ddof=1)) if len(column) > 1 else float("nan")
            )
        rows.append(entry)
    return pd.DataFrame(rows)


def ratios(comparison: pd.DataFrame, metrics: tuple[str, ...], grain: tuple[str, ...]):
    """SIT over RevIN per (*grain, mode), one number per metric.

    Kept as its own table rather than a third column beside each pair, so the
    disaggregated tables stay one-number-per-cell.
    """
    present = [
        name
        for base in metrics
        for name in (base, f"{base}_median")
        if name in comparison.columns
    ]
    keys = ("model", *grain, "mode")
    rows = []
    for values, group in comparison.groupby(list(keys)):
        entry = dict(zip(keys, values))
        indexed = group.set_index("space")
        for name in present:
            if {"SIT", "RevIN"} <= set(indexed.index):
                revin = indexed.loc["RevIN", name]
                entry[name] = (
                    float(indexed.loc["SIT", name] / revin)
                    if pd.notna(revin) and revin
                    else float("nan")
                )
        rows.append(entry)
    return pd.DataFrame(rows)


def render_report(
    comparison: pd.DataFrame,
    ratio_table: pd.DataFrame,
    per_run: pd.DataFrame,
    grain: tuple[str, ...],
    title: str,
) -> str:
    """Three tables: weighted mean, subset median, and the SIT/RevIN ratio.

    Rows are disaggregated, one condition each, and every cell holds a single
    number. Metrics are the columns, so a benchmark reads across and a
    condition reads down.
    """

    def block(frame, columns, keys, label, note):
        present = [name for name in columns if name in frame.columns]
        present = [name for name in present if frame[name].notna().any()]
        if not present:
            return []
        # M1, M3, and Tourism have no series long enough for `fixed`, so those
        # rows carry nothing. Dropping them keeps the table to modes that
        # actually produced a number.
        frame = frame[frame[present].notna().any(axis=1)]
        if frame.empty:
            return []
        out = [f"## {label}", "", note, ""]
        out.append("| " + " | ".join((*keys, *present)) + " |")
        out.append("|" + "|".join(["---"] * (len(keys) + len(present))) + "|")
        for _, row in frame.sort_values(list(keys)).iterrows():
            cells = [str(row[key]) for key in keys]
            for name in present:
                value = row[name]
                cells.append("" if pd.isna(value) else f"{value:.4f}")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
        return out

    metrics = ACCURACY_METRICS + STABILITY_METRICS
    medians = tuple(f"{name}_median" for name in metrics)
    keys = ("model", *grain, "mode", "space")

    lines = [f"# {title}", ""]
    runs = per_run[["model", "condition", "seed"]].drop_duplicates()
    lines.append(f"{len(runs)} runs scored, {per_run['seed'].nunique()} seeds.")
    lines.append("")
    lines.append("Every metric is lower-is-better. Values are the mean over seeds.")
    lines.append("")

    lines += block(
        comparison,
        metrics,
        keys,
        "Subset-weighted mean",
        "Subsets weighted by series count within each benchmark.",
    )
    lines += block(
        comparison,
        medians,
        keys,
        "Subset median",
        "Median across subsets, unweighted. Prefer this for nMSE, whose "
        "per-series denominator is unbounded and whose mean a single "
        "degenerate subset can dominate.",
    )
    lines += block(
        ratio_table,
        metrics + medians,
        ("model", *grain, "mode"),
        "SIT / RevIN",
        "Below 1 favors SIT.",
    )
    return "\n".join(lines)


# The deliverables. The main paper reports one row per benchmark; the appendix
# gets the same comparison by frequency, both pooled across benchmarks and
# broken out within each. The pooled-by-frequency view weights subsets by
# series count like the others, so M4's 100,000 series dominate any frequency
# it appears in. Read it as a trend across frequencies, not as a per-benchmark
# claim.
GRAINS = {
    "main": (("suite",), "TSFM held-out evaluation by benchmark"),
    "frequency": (("freq",), "TSFM held-out evaluation by frequency"),
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
        ratio_table = ratios(comparison, metrics, grain)
        per_run.to_csv(out_dir / f"eval_by_run_{label}.csv", index=False)
        comparison.to_csv(out_dir / f"eval_comparison_{label}.csv", index=False)
        ratio_table.to_csv(out_dir / f"eval_ratio_{label}.csv", index=False)
        report = render_report(comparison, ratio_table, per_run, grain, title)
        (out_dir / f"eval_report_{label}.md").write_text(report)
        print(f"\n{report}")
        print(f"wrote {out_dir / f'eval_comparison_{label}.csv'}")
        print(f"wrote {out_dir / f'eval_ratio_{label}.csv'}")
        print(f"wrote {out_dir / f'eval_report_{label}.md'}")


if __name__ == "__main__":
    main()

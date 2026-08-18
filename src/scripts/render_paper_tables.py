"""Renders the held-out TSFM evaluation as the paper's LaTeX tables.

One table per grain, following overleaf/PLAN.md: rows are benchmarks (or
benchmark by frequency), columns are models split into the scale-invariant
and scale-contaminated loss spaces, with the fitted classical references in
their own block on the right.

The paper's vocabulary is not the harness's. `collect_tsfm_eval` labels the
two conditions SIT and RevIN; here they become scale-invariant and
scale-contaminated, per overleaf/SKILL.md.

  uv run python -m src.scripts.render_paper_tables \
      --zero-shot chronos2=<dir> moirai2=<dir> \
      --in-domain chronos2=<dir> moirai2=<dir> \
      --baselines <dir> --output-dir overleaf/extended_draft/tables
"""

import argparse
from pathlib import Path

import pandas as pd

from src.scripts import collect_tsfm_eval as collect

# Mean for MASE and WQL, median across subsets for nMSE. nMSE's per-series
# denominator is the context variance and is unbounded, so one degenerate
# subset dominates its mean; see `collect_tsfm_eval.collapse`.
METRICS = (
    ("mase", "MASE", "mean"),
    ("wql", "WQL", "mean"),
    ("nmse", "nMSE", "median"),
)
SPACES = (("SIT", "Inv."), ("RevIN", "Contam."))
BASELINES = (
    ("seasonal_naive", r"\SNaive"),
    ("ets", r"\ETS"),
    ("arima", r"\ARIMA"),
)
SUITE_LABELS = {
    "m1": "M1",
    "m3": "M3",
    "m4": "M4",
    "tourism": "Tourism",
    "favorita": "Favorita",
    "gifteval": "GIFT-Eval",
}
SUITE_ORDER = ("m1", "m3", "m4", "tourism", "favorita", "gifteval")
# Folded so the gluonts aliases do not split a frequency across two rows.
FREQ_ALIAS = {"W-SUN": "W", "Q-DEC": "Q", "A-DEC": "Y", "A": "Y"}
FREQ_ORDER = ("Y", "Q", "M", "W", "D", "H", "T", "S", "15T", "10T", "30T", "5T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-shot", nargs="+", required=True)
    parser.add_argument("--in-domain", nargs="+", required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_model_runs(specs: list[str]) -> pd.DataFrame:
    """Concatenates the per-run subset tables for each `model=dir` spec."""
    frames = []
    for spec in specs:
        model, _, directory = spec.partition("=")
        if not directory:
            raise ValueError(f"expected model=dir, got {spec!r}")
        frame = collect.load_runs(Path(directory))
        frame["model"] = model
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_baselines(root: Path) -> pd.DataFrame:
    frames = []
    for name, _ in BASELINES:
        path = root / name / "eval_by_subset.csv"
        if not path.is_file():
            raise FileNotFoundError(f"no baseline table at {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def fold_frequency(table: pd.DataFrame) -> pd.DataFrame:
    return table.assign(freq=table["freq"].replace(FREQ_ALIAS))


def summarize(table: pd.DataFrame, grain: tuple[str, ...]) -> pd.DataFrame:
    """Native-mode results at `grain`, one row per model and loss space.

    Native is the only mode reported: it is each suite's own published
    protocol, and it is the only mode every suite has, `fixed` yielding
    nothing at all on M1, M3, and Tourism.
    """
    native = table[table["mode"] == "native"]
    per_run = collect.collapse(native, collect.ACCURACY_METRICS, grain)
    return collect.compare_conditions(per_run, collect.ACCURACY_METRICS, grain)


def cell(frame: pd.DataFrame, key: dict, column: str) -> str:
    mask = pd.Series(True, index=frame.index)
    for name, value in key.items():
        mask &= frame[name] == value
    rows = frame[mask]
    if rows.empty or pd.isna(rows.iloc[0][column]):
        return "--"
    return f"{rows.iloc[0][column]:.3f}"


def best_model(values: list[str], n_model_cols: int) -> list[str]:
    """Bolds the lowest of the model columns, every metric being lower-is-better.

    Only the model columns compete. The reference estimators are fitted on
    the series they forecast, so bolding one of them as the row winner would
    read as a like-for-like loss that the setup does not support.
    """
    numeric = [(i, float(v)) for i, v in enumerate(values[:n_model_cols]) if v != "--"]
    if not numeric:
        return values
    winner = min(numeric, key=lambda pair: pair[1])[0]
    out = list(values)
    out[winner] = rf"\textbf{{{out[winner]}}}"
    return out


def render(
    models: pd.DataFrame,
    baselines: pd.DataFrame,
    grain: tuple[str, ...],
    row_keys: list[tuple],
    row_labels: list[list[str]],
    model_order: list[tuple[str, str]],
    caption: str,
    label: str,
) -> str:
    n_model_cols = 2 * len(model_order)
    column_spec = "l" * len(grain) + "".join("cc" for _ in model_order) + "ccc"

    header_models = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{tex}}}" for _, tex in model_order
    )
    cmid_start = len(grain) + 1
    cmids = []
    for i in range(len(model_order)):
        left = cmid_start + 2 * i
        cmids.append(rf"\cmidrule(lr){{{left}-{left + 1}}}")
    ref_left = cmid_start + n_model_cols
    cmids.append(rf"\cmidrule(lr){{{ref_left}-{ref_left + 2}}}")

    sub = " & ".join(short for _ in model_order for _, short in SPACES)
    refs = " & ".join(tex for _, tex in BASELINES)

    lines = [
        r"\begin{table}[t!]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\resizebox{\textwidth}{!}{",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join([""] * len(grain))
        + f" & {header_models} & "
        + r"\multicolumn{3}{c}{Reference} \\",
        "".join(cmids),
        " & ".join(r"\textbf{" + h + "}" for h in grain_headers(grain))
        + f" & {sub} & {refs} "
        + r"\\",
    ]

    for metric, metric_label, statistic in METRICS:
        column = metric if statistic == "mean" else f"{metric}_median"
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{len(grain) + n_model_cols + 3}}}{{l}}"
            rf"{{\emph{{{metric_label}}}}} \\"
        )
        for key, labels in zip(row_keys, row_labels):
            selector = dict(zip(grain, key))
            values = []
            for model, _ in model_order:
                for space, _ in SPACES:
                    values.append(
                        cell(
                            models, {**selector, "model": model, "space": space}, column
                        )
                    )
            for name, _ in BASELINES:
                values.append(cell(baselines, {**selector, "model": name}, column))
            lines.append(" & ".join(labels + best_model(values, n_model_cols)) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}", ""]
    return "\n".join(lines)


def grain_headers(grain: tuple[str, ...]) -> list[str]:
    return ["Benchmark" if name == "suite" else "Freq." for name in grain]


def suite_rows(frame: pd.DataFrame) -> tuple[list[tuple], list[list[str]]]:
    present = [s for s in SUITE_ORDER if s in set(frame["suite"])]
    return [(s,) for s in present], [[SUITE_LABELS[s]] for s in present]


def suite_frequency_rows(frame: pd.DataFrame) -> tuple[list[tuple], list[list[str]]]:
    keys, labels = [], []
    for suite in SUITE_ORDER:
        sub = frame[frame["suite"] == suite]
        if sub.empty:
            continue
        freqs = sorted(
            sub["freq"].dropna().unique(),
            key=lambda f: FREQ_ORDER.index(f) if f in FREQ_ORDER else len(FREQ_ORDER),
        )
        for i, freq in enumerate(freqs):
            keys.append((suite, freq))
            labels.append([SUITE_LABELS[suite] if i == 0 else "", freq])
    return keys, labels


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_runs = fold_frequency(load_baselines(args.baselines))
    model_order = [("chronos2", r"\Chronostwo"), ("moirai2", r"\Moiraitwo")]

    space_note = (
        "Inv. is the scale-invariant loss and Contam. the scale-contaminated "
        "one, the only difference between the two runs of a model. "
        "MASE and WQL are subset-weighted means; nMSE is the median across "
        "subsets, its per-series denominator being unbounded. "
        "Lower is better throughout; the better loss space in each row is bold. "
        "The reference estimators are fitted on the series they forecast, so "
        "they are not zero-shot and are separated accordingly."
    )

    for name, specs, setting in (
        ("tsfm_heldout_zeroshot", args.zero_shot, "zero-shot"),
        ("tsfm_heldout_indomain", args.in_domain, "in-domain"),
    ):
        runs = fold_frequency(load_model_runs(specs))
        seeds = int(runs["seed"].nunique())
        if setting == "zero-shot":
            preamble = (
                "Held-out evaluation of models pretrained on GiftEvalPretrain "
                "with M1, M3, Tourism, and Favorita removed, so every benchmark "
                "is zero-shot."
            )
        else:
            preamble = (
                "The same comparison after the held-out suites' training "
                "regions are added to the pretraining corpus. These models are "
                "no longer zero-shot on M1, M3, M4, Tourism, or Favorita."
            )

        by_suite = summarize(runs, ("suite",))
        base_suite = summarize(baseline_runs, ("suite",))
        keys, labels = suite_rows(by_suite)
        (args.output_dir / f"{name}_main.tex").write_text(
            render(
                by_suite,
                base_suite,
                ("suite",),
                keys,
                labels,
                model_order,
                f"{preamble} Means over {seeds} seeds. {space_note}",
                f"tab:{name}-main",
            )
        )

        by_freq = summarize(runs, ("suite", "freq"))
        base_freq = summarize(baseline_runs, ("suite", "freq"))
        keys, labels = suite_frequency_rows(by_freq)
        (args.output_dir / f"{name}_by_frequency.tex").write_text(
            render(
                by_freq,
                base_freq,
                ("suite", "freq"),
                keys,
                labels,
                model_order,
                f"{preamble} Broken out by frequency. Means over {seeds} seeds. "
                f"{space_note}",
                f"tab:{name}-by-frequency",
            )
        )
        print(f"wrote {name}_main.tex and {name}_by_frequency.tex")


if __name__ == "__main__":
    main()

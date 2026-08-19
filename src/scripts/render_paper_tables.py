"""Renders the held-out TSFM evaluation as the paper's LaTeX tables.

One table per grain, following overleaf/PLAN.md: rows are benchmarks (or
benchmark by frequency), columns are models split into the scale-invariant
and scale-contaminated loss spaces, with the fitted classical references in
their own block on the right.

The paper's vocabulary is not the harness's. `collect_tsfm_eval` labels the
two conditions SIT and RevIN. Here they become scale-invariant and
scale-contaminated, per overleaf/SKILL.md.

  uv run python -m src.scripts.render_paper_tables \
      --zero-shot chronos2=<dir> moirai2=<dir> \
      --in-domain chronos2=<dir> moirai2=<dir> \
      --baselines <dir> --output-dir overleaf/extended_draft/tables
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from src.scripts import collect_tsfm_eval as collect

# Mean for MASE and WQL, median across subsets for nMSE. nMSE's per-series
# denominator is the context variance and is unbounded, so one degenerate
# subset dominates its mean. See `collect_tsfm_eval.collapse`.
METRICS = (
    ("mase", "MASE", "mean"),
    ("wql", "WQL", "mean"),
)
SPACES = (("SIT", r"\oursLag"), ("RevIN", r"\SCL"))
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
FREQ_ORDER = ("Y", "Q", "M", "W", "D", "H", "T", "S")
STAR = r"$^{*}$"


def base_frequency(freq: str) -> str:
    """The bare frequency behind a gluonts alias.

    GIFT-Eval carries the multiplier and the anchor in the code, so weekly
    data arrives as five distinct strings (`W-SUN` through `W-FRI`) and
    minutely as three (`5T`, `10T`, `15T`). Left alone each becomes its own
    row, which splits one frequency across five lines of the table and makes
    the weekly result unreadable. Stripping the multiplier and the anchor
    leaves the eight base codes the paper actually talks about.
    """
    bare = re.sub(r"^\d+", "", str(freq)).split("-")[0]
    return "Y" if bare == "A" else bare


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


def checkpoint_steps(specs: list[str]) -> int:
    """The update count behind a set of eval directories.

    Read from each eval's own hydra config rather than passed in, so a table
    can never claim a step count the checkpoints it summarizes do not have.
    Every run in one table must share it, which is what makes a single number
    in the caption honest.
    """
    steps = set()
    for spec in specs:
        directory = Path(spec.partition("=")[2])
        for config in sorted(directory.glob("*/.hydra/config.yaml")):
            checkpoint = yaml.safe_load(config.read_text())["checkpoint"]
            match = re.search(r"checkpoint_step(\d+)\.pt$", checkpoint)
            if match is None:
                raise ValueError(f"cannot read a step count from {checkpoint}")
            steps.add(int(match.group(1)))
    if len(steps) != 1:
        raise ValueError(f"eval directories mix checkpoint steps {sorted(steps)}")
    return steps.pop()


def load_baselines(root: Path) -> pd.DataFrame:
    frames = []
    for name, _ in BASELINES:
        path = root / name / "eval_by_subset.csv"
        if not path.is_file():
            raise FileNotFoundError(f"no baseline table at {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def fold_frequency(table: pd.DataFrame) -> pd.DataFrame:
    return table.assign(freq=table["freq"].map(base_frequency))


def collapsed_frequencies(table: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    """Which (benchmark, base frequency) cells pool more than their own code.

    Keyed so the star lands only where something was actually pooled. M4's
    weekly data is already `W` and gets no star. GIFT-Eval's is five anchored
    codes and does.
    """
    out: dict[tuple[str, str], list[str]] = {}
    for (suite, freq), group in table.groupby(["suite", "freq"]):
        del group
        out.setdefault((suite, base_frequency(freq)), []).append(str(freq))
    return {key: sorted(values) for key, values in out.items() if values != [key[1]]}


def summarize_baselines(table: pd.DataFrame, grain: tuple[str, ...]) -> pd.DataFrame:
    """Native-mode reference results at `grain`, one row per estimator.

    Deliberately not routed through `compare_conditions`. That function reads
    a loss space out of the condition name, and a baseline has no loss space:
    every one of them would come back labelled RevIN because its name does
    not end in "normalized". The numbers survive that only because there is
    exactly one row per estimator to pick from, which is an accident rather
    than a guarantee.
    """
    native = table[table["mode"] == "native"]
    per_run = collapse_runs(native, grain)
    keys = ["model", *grain]
    aggregated = per_run.groupby(keys, as_index=False).mean(numeric_only=True)
    counts = per_run.groupby(keys, as_index=False).size()
    merged = aggregated.merge(counts, on=keys)
    if (merged["size"] != 1).any():
        raise ValueError("a reference estimator has more than one run per cell")
    return merged.drop(columns="size")


def collapse_runs(table: pd.DataFrame, grain: tuple[str, ...]) -> pd.DataFrame:
    return collect.collapse(table, collect.ACCURACY_METRICS, grain)


def summarize(table: pd.DataFrame, grain: tuple[str, ...]) -> pd.DataFrame:
    """Native-mode results at `grain`, one row per model and loss space.

    Native is the only mode reported: it is each suite's own published
    protocol, and it is the only mode every suite has, `fixed` yielding
    nothing at all on M1, M3, and Tourism.
    """
    native = table[table["mode"] == "native"]
    per_run = collect.collapse(native, collect.ACCURACY_METRICS, grain)
    return collect.compare_conditions(per_run, collect.ACCURACY_METRICS, grain)


def value(frame: pd.DataFrame, key: dict, column: str) -> float | None:
    mask = pd.Series(True, index=frame.index)
    for name, item in key.items():
        mask &= frame[name] == item
    rows = frame[mask]
    if rows.empty or pd.isna(rows.iloc[0][column]):
        return None
    return float(rows.iloc[0][column])


def fmt(number: float | None) -> str:
    return "--" if number is None else f"{number:.3f}"


def delta(sit: float | None, revin: float | None) -> str:
    """SIT against RevIN as a percentage, coloured by which one won.

    Every metric here is lower-is-better, so a negative delta is a win for
    the scale-invariant loss. The sign carries that on its own. The colour is
    there so a reader can scan a column of forty rows without doing the
    arithmetic.
    """
    if sit is None or revin is None or revin == 0.0:
        return "--"
    change = 100.0 * (sit / revin - 1.0)
    macro = "ourswin" if change < 0 else "oursloss"
    return rf"\{macro}{{{change:+.1f}}}"


def render(
    models: pd.DataFrame,
    baselines: pd.DataFrame,
    grain: tuple[str, ...],
    row_keys: list[tuple],
    row_labels: list[list[str]],
    model_order: list[tuple[str, str]],
    caption: str,
    label: str,
    size: str,
    metrics: tuple = METRICS,
) -> str:
    """One table. Each architecture gets SIT, RevIN, and their delta.

    The reference block sits behind a double rule because those estimators
    are fitted on the series they forecast. The rule is the visual form of
    the same caveat the caption states.
    """
    n_lead = len(grain)
    per_model = len(SPACES) + 1
    n_model_cols = per_model * len(model_order)
    column_spec = (
        "l" * n_lead
        + "".join("c" * per_model for _ in model_order)
        + "||"
        + "c" * len(BASELINES)
    )

    header_models = " & ".join(
        rf"\multicolumn{{{per_model}}}{{c}}{{{tex}}}" for _, tex in model_order
    )
    cmids = []
    for i in range(len(model_order)):
        left = n_lead + 1 + per_model * i
        cmids.append(rf"\cmidrule(lr){{{left}-{left + per_model - 1}}}")
    ref_left = n_lead + 1 + n_model_cols
    cmids.append(rf"\cmidrule(lr){{{ref_left}-{ref_left + len(BASELINES) - 1}}}")

    sub = " & ".join(
        part
        for _ in model_order
        for part in ([short for _, short in SPACES] + [r"$\Delta\%$"])
    )
    refs = " & ".join(tex for _, tex in BASELINES)

    lines = [
        "% Generated by src/scripts/render_paper_tables.py. Do not edit by hand.",
        r"\begin{table}[t!]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.15}",
        rf"\{size}",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join([""] * n_lead)
        + f" & {header_models} & "
        + rf"\multicolumn{{{len(BASELINES)}}}{{c}}{{Reference}} \\",
        "".join(cmids),
        " & ".join(r"\textbf{" + h + "}" for h in grain_headers(grain))
        + f" & {sub} & {refs} "
        + r"\\",
    ]

    total_cols = n_lead + n_model_cols + len(BASELINES)
    for metric, metric_label, statistic in metrics:
        column = metric if statistic == "mean" else f"{metric}_median"
        lines.append(r"\midrule")
        # A single-metric table names its metric in the caption, so a banner
        # row would only repeat it.
        if len(metrics) > 1:
            lines.append(
                rf"\multicolumn{{{total_cols}}}{{l}}{{\emph{{{metric_label}}}}} \\"
            )
        for key, labels in zip(row_keys, row_labels):
            selector = dict(zip(grain, key))
            cells = []
            for model, _ in model_order:
                pair = [
                    value(models, {**selector, "model": model, "space": space}, column)
                    for space, _ in SPACES
                ]
                shown = [fmt(number) for number in pair]
                # Bolded within the architecture, so each model is read as its
                # own SIT-against-RevIN comparison rather than against the
                # other architecture.
                present = [(i, x) for i, x in enumerate(pair) if x is not None]
                if present:
                    winner = min(present, key=lambda item: item[1])[0]
                    shown[winner] = rf"\textbf{{{shown[winner]}}}"
                cells.extend(shown + [delta(pair[0], pair[1])])
            for name, _ in BASELINES:
                cells.append(fmt(value(baselines, {**selector, "model": name}, column)))
            lines.append(" & ".join(labels + cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def grain_headers(grain: tuple[str, ...]) -> list[str]:
    return ["Benchmark" if name == "suite" else "Freq." for name in grain]


def suite_rows(frame: pd.DataFrame) -> tuple[list[tuple], list[list[str]]]:
    present = [s for s in SUITE_ORDER if s in set(frame["suite"])]
    return [(s,) for s in present], [[SUITE_LABELS[s]] for s in present]


def suite_frequency_rows(
    frame: pd.DataFrame, collapsed: dict[tuple[str, str], list[str]]
) -> tuple[list[tuple], list[list[str]]]:
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
            star = STAR if (suite, freq) in collapsed else ""
            labels.append([SUITE_LABELS[suite] if i == 0 else "", f"{freq}{star}"])
    return keys, labels


def collapse_note(collapsed: dict[tuple[str, str], list[str]]) -> str:
    """Explain the star used for pooled GIFT-Eval frequency aliases."""
    if not collapsed:
        return ""
    return (
        "A star marks a base frequency that pools anchored or multiplied "
        "GIFT-Eval aliases. "
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_baselines = load_baselines(args.baselines)
    baseline_runs = fold_frequency(raw_baselines)
    model_order = [("chronos2", r"\Chronostwo"), ("moirai2", r"\Moiraitwo")]

    space_note = (
        r"\oursLag uses scale-invariant loss and \SCL uses "
        "scale-contaminated loss. These are the only differences between "
        "the two runs of each architecture. "
        r"$\Delta\%$ reports \oursLag relative to \SCL. "
        r"\ourswin{Green} denotes an improvement and \oursloss{red} a regression. "
        "The better loss is bold within each architecture. "
        "References are fitted on the series they forecast and appear behind "
        "the double rule."
    )

    for name, specs, setting in (
        ("tsfm_heldout_zeroshot", args.zero_shot, "zero-shot"),
        ("tsfm_heldout_indomain", args.in_domain, "in-domain"),
    ):
        raw_runs = load_model_runs(specs)
        runs = fold_frequency(raw_runs)
        collapsed = collapsed_frequencies(
            pd.concat([raw_runs, raw_baselines], ignore_index=True)
        )
        seeds = int(runs["seed"].nunique())
        steps = checkpoint_steps(specs)
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
        base_suite = summarize_baselines(baseline_runs, ("suite",))
        keys, labels = suite_rows(by_suite)
        (args.output_dir / f"{name}_main.tex").write_text(
            render(
                by_suite,
                base_suite,
                ("suite",),
                keys,
                labels,
                model_order,
                f"{preamble} Both models are pretrained for {steps:,} updates at batch "
                f"size 512. Means over {seeds} seeds. {space_note}",
                f"tab:{name}-main",
                "small",
            )
        )

        by_freq = summarize(runs, ("suite", "freq"))
        base_freq = summarize_baselines(baseline_runs, ("suite", "freq"))
        keys, labels = suite_frequency_rows(by_freq, collapsed)
        note = collapse_note(collapsed)
        # One file per metric. Both metrics in one table is 64 body rows,
        # which overruns a page even at scriptsize, and longtable is not in
        # the draft's preamble.
        for metric in METRICS:
            setting_label = "Zero-shot" if setting == "zero-shot" else "In-domain"
            (args.output_dir / f"{name}_by_frequency_{metric[0]}.tex").write_text(
                render(
                    by_freq,
                    base_freq,
                    ("suite", "freq"),
                    keys,
                    labels,
                    model_order,
                    f"{setting_label} {metric[1]} by benchmark and frequency "
                    f"after {steps:,} updates, averaged over {seeds} seeds. {note}"
                    f"{space_note}",
                    f"tab:{name}-by-frequency-{metric[0]}",
                    "scriptsize",
                    (metric,),
                )
            )
        print(
            f"wrote {name}_main.tex and {name}_by_frequency_*.tex "
            f"({steps} updates, {seeds} seeds)"
        )


if __name__ == "__main__":
    main()

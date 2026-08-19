"""Generate supervised M-series LaTeX tables."""

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = {
    "nhits": r"\model{N-HiTS}",
    "nbeats": r"\model{N-BEATS}",
    "patchtst": r"\PatchTST",
}
FREQUENCIES = ("Y", "Q", "M", "W", "D", "H")
BENCHMARKS = ("m1", "m3", "m4", "tourism")
LABELS = {"m1": "M1", "m3": "M3", "m4": "M4", "tourism": "Tourism"}
BENCHMARK_FREQUENCIES = {
    "m1": ("Y", "Q", "M"),
    "m3": ("Y", "Q", "M"),
    "m4": FREQUENCIES,
    "tourism": ("Y", "Q", "M"),
}
CONDITIONS = ("sit", "revin")
STATISTICS = ("standard", "causal")
METRICS = ("model_wql", "model_mase")
GENERATOR = "src/plotting/scripts/generate_supervised_mseries_tables.py"


def load_results(root: Path):
    expected = {
        (m, f, c, s)
        for m in MODELS
        for f in FREQUENCIES
        for c in CONDITIONS
        for s in STATISTICS
    }
    results = {}
    for path in sorted(root.glob("*/*/*/*/seed0/metrics_by_benchmark.json")):
        run = json.loads(path.read_text())
        key = (run["model"], run["frequency"], run["condition"], run["normalization"])
        if key in results:
            raise ValueError(f"duplicate result {key}")
        results[key] = run
    if set(results) != expected:
        raise ValueError(f"missing {sorted(expected - set(results))}")
    return results


def load_baselines(root: Path):
    shard_counts = {("m4", "Y"): 8, ("m4", "Q"): 23, ("m4", "M"): 33, ("m4", "D"): 5}
    expected = {
        (name, benchmark, frequency, shard)
        for name in ("ets", "arima")
        for benchmark in BENCHMARKS
        for frequency in BENCHMARK_FREQUENCIES[benchmark]
        for shard in range(shard_counts.get((benchmark, frequency), 1))
    }
    outputs = [json.loads(path.read_text()) for path in sorted(root.glob("*.json"))]
    found = {
        (x["baseline"], x["suite"], x["frequency"], x["shard_index"]) for x in outputs
    }
    if found != expected:
        raise ValueError(f"missing {sorted(expected - found)}")
    return outputs


def ratio(rows, prefix=""):
    stem = f"{prefix}_" if prefix else ""
    numerator = sum(row[f"{stem}wql_num"] for row in rows)
    denominator = sum(row[f"{stem}wql_den"] for row in rows)
    if denominator <= 0:
        raise ValueError("non-positive WQL denominator")
    return numerator / denominator


def model_wql(results, model, condition, statistics, benchmark, frequency):
    rows = []
    for current in (frequency,) if frequency else FREQUENCIES:
        rows += [
            row
            for row in results[(model, current, condition, statistics)]["test_origins"]
            if benchmark is None or row["subset"] == benchmark
        ]
    return ratio(rows, "model")


def reference_wql(results, baselines, name, benchmark, frequency):
    if name == "seasonal_naive":
        rows = []
        for current in (frequency,) if frequency else FREQUENCIES:
            rows += [
                row
                for row in results[("nhits", current, "sit", "standard")][
                    "test_origins"
                ]
                if benchmark is None or row["subset"] == benchmark
            ]
        return ratio(rows, "seasonal_naive")
    rows = [
        row
        for output in baselines
        if output["baseline"] == name
        and (benchmark is None or output["suite"] == benchmark)
        and (frequency is None or output["frequency"] == frequency)
        for row in output["rows"]
    ]
    return ratio(rows)


def shown(value, bold=False):
    text = f"{value:.3f}"
    return rf"\textbf{{{text}}}" if bold else text


def grain_rows(grain):
    if grain == "frequency":
        return [(None, f) for f in FREQUENCIES], ("Frequency",)
    if grain == "benchmark":
        return [(b, None) for b in BENCHMARKS], ("Benchmark",)
    if grain == "benchmark_frequency":
        return [(b, f) for b in BENCHMARKS for f in BENCHMARK_FREQUENCIES[b]], (
            "Benchmark",
            "Frequency",
        )
    raise ValueError(grain)


def render_wql_table(results, baselines, grain):
    rows, headers = grain_rows(grain)
    labels = {
        "frequency": "supervised-mseries-wql",
        "benchmark": "supervised-mseries-benchmark-wql",
        "benchmark_frequency": "supervised-mseries-benchmark-frequency-wql",
    }
    descriptions = {
        "frequency": "frequency",
        "benchmark": "benchmark",
        "benchmark_frequency": "benchmark and frequency",
    }
    caption = (
        f"Aggregate WQL by {descriptions[grain]} in the supervised M-series "
        r"experiment. \SCLLag and \SCLCausal use the scale-contaminated loss "
        r"with lag and causal scaling statistics. \oursLag and \oursCausal use "
        "the corresponding scale-invariant loss. Lower is better, and the "
        "better loss is bold within each matched pair. References are fitted "
        "at every rolling origin using only available history."
    )
    column_spec = "l" * (len(headers) + 1) + "cccc||ccc"
    padding = "3pt" if grain == "benchmark_frequency" else "4pt"
    lines = [
        f"% Generated by {GENERATOR}. Do not edit by hand.",
        r"\begin{table}[t!]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{labels[grain]}}}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\scriptsize",
        rf"\setlength{{\tabcolsep}}{{{padding}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        r"\textbf{Architecture} & "
        + " & ".join(rf"\textbf{{{header}}}" for header in headers)
        + r" & \SCLLag & \oursLag & \SCLCausal & \oursCausal"
        + r" & \SNaive & \ETS & \ARIMA \\",
        r"\midrule",
    ]
    for model_index, (model, model_label) in enumerate(MODELS.items()):
        previous = None
        for row_index, (benchmark, frequency) in enumerate(rows):
            scl_lag = model_wql(
                results, model, "revin", "standard", benchmark, frequency
            )
            ours_lag = model_wql(
                results, model, "sit", "standard", benchmark, frequency
            )
            scl_causal = model_wql(
                results, model, "revin", "causal", benchmark, frequency
            )
            ours_causal = model_wql(
                results, model, "sit", "causal", benchmark, frequency
            )
            cells = [
                shown(scl_lag, scl_lag <= ours_lag),
                shown(ours_lag, ours_lag < scl_lag),
                shown(scl_causal, scl_causal <= ours_causal),
                shown(ours_causal, ours_causal < scl_causal),
            ]
            cells += [
                shown(reference_wql(results, baselines, name, benchmark, frequency))
                for name in ("seasonal_naive", "ets", "arima")
            ]
            if grain == "frequency":
                row_labels = [frequency]
            elif grain == "benchmark":
                row_labels = [LABELS[benchmark]]
            else:
                row_labels = [
                    LABELS[benchmark] if benchmark != previous else "",
                    frequency,
                ]
                previous = benchmark
            architecture = (
                rf"\multirow{{{len(rows)}}}{{*}}{{{model_label}}}"
                if row_index == 0
                else ""
            )
            lines.append(" & ".join([architecture, *row_labels, *cells]) + r" \\")
        if model_index < len(MODELS) - 1:
            lines.append(r"\midrule")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def paired_effect(results, model, metric, comparison, fixed_condition):
    ratios = []
    for frequency in FREQUENCIES:
        if comparison == "causal_standard":
            numerator = results[(model, frequency, fixed_condition, "causal")][
                "test_aggregate"
            ][metric]
            denominator = results[(model, frequency, fixed_condition, "standard")][
                "test_aggregate"
            ][metric]
        elif comparison == "sit_revin":
            numerator = results[(model, frequency, "sit", fixed_condition)][
                "test_aggregate"
            ][metric]
            denominator = results[(model, frequency, "revin", fixed_condition)][
                "test_aggregate"
            ][metric]
        else:
            raise ValueError(comparison)
        ratios.append(numerator / denominator)
    return float((np.exp(np.mean(np.log(ratios))) - 1) * 100), sum(
        x < 1 for x in ratios
    )


def render_effect_table(results):
    lines = [
        f"% Generated by {GENERATOR}. Do not edit by hand.",
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Paired effects across the six supervised frequencies. "
            r"\SCL uses the scale-contaminated loss and \ours uses the "
            r"scale-invariant loss. Each cell shows geometric mean percentage "
            r"change and frequency wins in parentheses.}"
        ),
        r"\label{tab:supervised-mseries-effects}",
        r"\tiny",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        (
            r"& & \multicolumn{2}{c}{Causal / standard statistics} & "
            r"\multicolumn{2}{c}{\ours{} / \SCL} \\"
        ),
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Metric & Architecture & \ours & \SCL & Lag & Causal \\",
        r"\midrule",
    ]
    comparisons = (
        ("causal_standard", "sit"),
        ("causal_standard", "revin"),
        ("sit_revin", "standard"),
        ("sit_revin", "causal"),
    )
    for metric_index, metric in enumerate(METRICS):
        for model, model_label in MODELS.items():
            cells = []
            for comparison, fixed in comparisons:
                change, wins = paired_effect(results, model, metric, comparison, fixed)
                cells.append("$" + f"{change:+.1f}\\%$ ({wins}/6)")
            lines.append(
                f"{metric.removeprefix('model_').upper()} & {model_label} & "
                + " & ".join(cells)
                + r" \\"
            )
        if metric_index == 0:
            lines.append(r"\midrule")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    args = parser.parse_args()
    results = load_results(args.metrics_root)
    baselines = load_baselines(args.baseline_root)
    if not args.table_dir.is_dir():
        raise FileNotFoundError(args.table_dir)
    outputs = {
        "supervised_mseries_wql.tex": render_wql_table(results, baselines, "frequency"),
        "supervised_mseries_benchmark_wql.tex": render_wql_table(
            results, baselines, "benchmark"
        ),
        "supervised_mseries_benchmark_frequency_wql.tex": render_wql_table(
            results, baselines, "benchmark_frequency"
        ),
        "supervised_mseries_effects.tex": render_effect_table(results),
    }
    for filename, table in outputs.items():
        (args.table_dir / filename).write_text(table)


if __name__ == "__main__":
    main()

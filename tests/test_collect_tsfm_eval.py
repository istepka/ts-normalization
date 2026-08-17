"""The SIT-against-RevIN collector.

The property worth pinning is the weighting: benchmark rows weight their
subsets by series count, so a large subset must move the benchmark number
more than a small one. A flat mean over subsets would look plausible and be
wrong, which is the same failure that put pooled WQL 16% off GIFT-Eval.
"""

import pandas as pd
import pytest

from src.scripts import collect_tsfm_eval as collect

METRICS = collect.ACCURACY_METRICS + collect.STABILITY_METRICS


def _row(condition, seed, suite, subset, freq, n_series, mase):
    return {
        "model": "chronos2",
        "condition": condition,
        "seed": seed,
        "suite": suite,
        "mode": "native",
        "subset": subset,
        "freq": freq,
        "n_series": n_series,
        "mase": mase,
    }


@pytest.fixture
def table():
    """Two subsets of very different size, so flat and weighted means differ."""
    rows = []
    for seed in (0, 1):
        rows.append(
            _row("chronos2_normalized", seed, "m4", "m4_monthly", "M", 9000, 1.0)
        )
        rows.append(
            _row("chronos2_normalized", seed, "m4", "m4_hourly", "H", 1000, 2.0)
        )
        rows.append(_row("chronos2_original", seed, "m4", "m4_monthly", "M", 9000, 2.0))
        rows.append(_row("chronos2_original", seed, "m4", "m4_hourly", "H", 1000, 4.0))
    return pd.DataFrame(rows)


def test_benchmark_rows_weight_subsets_by_series_count(table):
    per_run = collect.collapse(table, METRICS, ("suite",))
    sit = per_run[per_run.condition == "chronos2_normalized"]
    # (9000 * 1.0 + 1000 * 2.0) / 10000 = 1.1, not the flat mean 1.5
    assert sit["mase"].unique() == pytest.approx(1.1)


def test_frequency_grain_keeps_the_subsets_apart(table):
    per_run = collect.collapse(table, METRICS, ("suite", "freq"))
    sit = per_run[per_run.condition == "chronos2_normalized"].set_index("freq")
    assert sit.loc["M", "mase"].unique() == pytest.approx(1.0)
    assert sit.loc["H", "mase"].unique() == pytest.approx(2.0)


def test_conditions_are_disaggregated_one_row_each(table):
    per_run = collect.collapse(table, METRICS, ("suite",))
    comparison = collect.compare_conditions(per_run, METRICS, ("suite",))
    assert set(comparison["space"]) == {"SIT", "RevIN"}
    indexed = comparison.set_index("space")
    assert indexed.loc["SIT", "mase"] == pytest.approx(1.1)
    assert indexed.loc["RevIN", "mase"] == pytest.approx(2.2)
    assert indexed.loc["SIT", "n_seeds"] == 2


def test_ratio_table_is_sit_over_revin(table):
    per_run = collect.collapse(table, METRICS, ("suite",))
    comparison = collect.compare_conditions(per_run, METRICS, ("suite",))
    row = collect.ratios(comparison, METRICS, ("suite",)).iloc[0]
    assert row["mase"] == pytest.approx(0.5)


def test_median_ignores_the_subset_weighting(table):
    """The mean weights m4_monthly's 9000 series; the median does not, and
    that difference is the whole point of reporting both."""
    per_run = collect.collapse(table, METRICS, ("suite",))
    sit = per_run[per_run.condition == "chronos2_normalized"].iloc[0]
    assert sit["mase"] == pytest.approx(1.1)
    assert sit["mase_median"] == pytest.approx(1.5)


def test_a_degenerate_subset_moves_the_mean_but_not_the_median():
    """The failure this exists to catch: on the first real run m4_daily
    carried nMSE 2.0e7 and dragged the M4 mean four orders of magnitude
    above the median, which tracked training-time eval."""
    rows = [
        _row("chronos2_normalized", 0, "m4", f"s{i}", "M", 100, 1.5) for i in range(9)
    ]
    rows.append(_row("chronos2_normalized", 0, "m4", "m4_daily", "D", 100, 2.0e7))
    per_run = collect.collapse(pd.DataFrame(rows), METRICS, ("suite",))
    assert per_run["mase"].iloc[0] > 1e6
    assert per_run["mase_median"].iloc[0] == pytest.approx(1.5)


def test_timesfm_native_original_is_read_as_revin():
    """TimesFM's original-space condition is not named `*_original`, so a
    naive equality check would silently file it under SIT."""
    rows = [
        _row("timesfm_normalized", 0, "m3", "m3_monthly", "M", 100, 1.0),
        _row("timesfm_native_original", 0, "m3", "m3_monthly", "M", 100, 2.0),
    ]
    per_run = collect.collapse(pd.DataFrame(rows), METRICS, ("suite",))
    comparison = collect.compare_conditions(per_run, METRICS, ("suite",))
    indexed = comparison.set_index("space")
    assert indexed.loc["SIT", "mase"] == pytest.approx(1.0)
    assert indexed.loc["RevIN", "mase"] == pytest.approx(2.0)


def test_report_tables_are_well_formed(table):
    per_run = collect.collapse(table, METRICS, ("suite", "freq"))
    comparison = collect.compare_conditions(per_run, METRICS, ("suite", "freq"))
    ratio_table = collect.ratios(comparison, METRICS, ("suite", "freq"))
    report = collect.render_report(
        comparison, ratio_table, per_run, ("suite", "freq"), "test report"
    )
    blocks = {}
    heading = None
    for line in report.splitlines():
        if line.startswith("## "):
            heading = line
            blocks[heading] = []
        elif line.startswith("|") and heading is not None:
            blocks[heading].append(line.count("|"))
    assert blocks, "no metric blocks rendered"
    for heading, widths in blocks.items():
        assert len(set(widths)) == 1, f"{heading} has ragged columns {set(widths)}"

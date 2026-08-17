"""Eval suite loaders.

The expensive suites (M4 at 100k series, Favorita, GIFT-Eval at 319k
instances) are verified against the real sources by
src/scripts/verify_eval_suites.py rather than here, with the results
recorded in notes/agentic_logs/2026-08-16-eval-harness.md. These tests cover
the parsing and splitting logic on synthetic inputs, plus the two real
checks that are cheap enough to run every time.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.eval import suites

GIFTEVAL_ROOT = Path("/zfsauton/scratch/istepka/lts/data/gift-eval")
MONASH_ROOT = Path("/zfsauton/scratch/istepka/lts/data/monash_eval")


def _write_tsf(path: Path, horizon: int, freq: str | None, rows: list[str]) -> None:
    header = [
        "# comment line",
        "@relation test",
        "@attribute series_name string",
    ]
    if freq is not None:
        header.append(f"@frequency {freq}")
    header += [f"@horizon {horizon}", "@missing false", "@data"]
    path.write_text("\n".join(header + rows) + "\n")


def test_read_tsf_parses_header_and_series(tmp_path):
    path = tmp_path / "x.tsf"
    _write_tsf(path, 2, "yearly", ["T1:1.0,2.0,3.0,4.0", "T2:5.0,6.0,7.0"])
    header, series = suites.read_tsf(path)

    assert header["horizon"] == "2"
    assert header["frequency"] == "yearly"
    assert [name for name, _ in series] == ["T1", "T2"]
    assert np.allclose(series[0][1], [1.0, 2.0, 3.0, 4.0])


def test_read_tsf_handles_a_start_timestamp_column(tmp_path):
    """M1/M3/Tourism carry a start timestamp between the name and the values,
    but M3 Other does not, so the parser must take the values from the last
    colon-separated field either way."""
    path = tmp_path / "x.tsf"
    _write_tsf(path, 1, "yearly", ["T1:1972-01-01 00-00-00:1.0,2.0,3.0"])
    _, series = suites.read_tsf(path)
    assert np.allclose(series[0][1], [1.0, 2.0, 3.0])


def test_monash_loader_splits_the_published_horizon_off_the_end(tmp_path, monkeypatch):
    directory = tmp_path / "m1_yearly"
    directory.mkdir()
    _write_tsf(directory / "d.tsf", 2, "yearly", ["T1:1,2,3,4,5", "T2:9,8,7,6"])
    monkeypatch.setitem(suites.EXPECTED_SERIES, "m1", 2)

    out = suites.load_monash(tmp_path, "m1")
    assert [s.item_id for s in out] == ["T1", "T2"]
    assert np.allclose(out[0].history, [1, 2, 3])
    assert np.allclose(out[0].actual, [4, 5])
    assert out[0].period == 1  # yearly has no shorter cycle


def test_monash_loader_rejects_a_series_with_no_history(tmp_path, monkeypatch):
    directory = tmp_path / "m1_yearly"
    directory.mkdir()
    _write_tsf(directory / "d.tsf", 4, "yearly", ["T1:1,2,3,4"])
    monkeypatch.setitem(suites.EXPECTED_SERIES, "m1", 1)
    with pytest.raises(ValueError, match="leaving no history"):
        suites.load_monash(tmp_path, "m1")


def test_m3_other_gets_no_seasonal_cycle(tmp_path, monkeypatch):
    """M3 Other has neither a frequency nor a start timestamp, so its
    seasonal period must be declared rather than derived."""
    directory = tmp_path / "m3_other"
    directory.mkdir()
    _write_tsf(directory / "d.tsf", 2, None, ["T1:1,2,3,4,5"])
    monkeypatch.setitem(suites.EXPECTED_SERIES, "m3", 1)

    out = suites.load_monash(tmp_path, "m3")
    assert out[0].period == 1
    assert out[0].freq == "Y"


def test_count_mismatch_is_an_error_not_a_silent_truncation(tmp_path):
    directory = tmp_path / "m1_yearly"
    directory.mkdir()
    _write_tsf(directory / "d.tsf", 1, "yearly", ["T1:1,2,3"])
    with pytest.raises(ValueError, match="expected 1001"):
        suites.load_monash(tmp_path, "m1")


def test_m4_csv_reader_drops_ragged_padding(tmp_path):
    path = tmp_path / "Yearly-train.csv"
    path.write_text('"V1","V2","V3","V4"\n"Y1","1","2","3"\n"Y2","4","5",\n')
    out = suites._read_m4_csv(path)
    assert np.allclose(out["Y1"], [1, 2, 3])
    assert np.allclose(out["Y2"], [4, 5])


def test_base_freq_code_strips_multipliers_and_anchors():
    assert suites.base_freq_code("A-DEC") == "A"
    assert suites.base_freq_code("10T") == "T"
    assert suites.base_freq_code("W-SUN") == "W"
    assert suites.base_freq_code("D") == "D"


def test_gluonts_seasonality_matches_the_benchmarks_convention():
    """GIFT-Eval scores through gluonts' DEFAULT_SEASONALITIES, where daily
    has no cycle and seconds take the hourly one. src/data/seasonality.py
    maps daily to 7 and seconds to the daily cycle for the training corpus,
    and using it here put our seasonal-naive MASE 1.5x to 5x off the
    benchmark's published numbers on every daily and 10-second config."""
    from src.data import seasonality as corpus_seasonality

    assert suites._gluonts_seasonality("D") == 1
    assert suites._gluonts_seasonality("10S") == 360
    assert corpus_seasonality.seasonal_period("D") == 7
    for freq, expected in [("H", 24), ("5T", 288), ("W-SUN", 1), ("A-DEC", 1)]:
        assert suites._gluonts_seasonality(freq) == expected, freq


def test_seasonal_period_travels_with_the_suite_not_the_frequency():
    """Two suites can score the same frequency on different periods, so the
    period is declared per suite rather than derived. M3 Other has no
    frequency at all to derive one from, and Favorita is our own definition
    which takes the weekly retail cycle where gluonts would use none."""
    assert suites.MONASH_PERIODS[None] == 1
    assert suites.FAVORITA_PERIOD == 7
    assert suites._gluonts_seasonality("D") == 1


def test_every_short_horizon_fits_the_models_native_window():
    """No autoregressive rollout is needed only because GIFT-Eval is
    restricted to the short term. The binding constraint is Moirai 2.0, whose
    native horizon is 64 (num_predict_token 4 * patch_size 16), not TimesFM's
    or Chronos-2's 128. The longest suite horizon is 60, so the margin is 4
    steps: any new suite, or a Moirai config with fewer predict tokens, needs
    rechecking here before it can be scored."""
    horizons = set(suites.GIFTEVAL_PRED_LENGTH.values())
    horizons |= set(suites.GIFTEVAL_M4_PRED_LENGTH.values())
    horizons |= {spec["horizon"] for spec in suites.M4_SPEC.values()}
    horizons.add(suites.FAVORITA_HORIZON)
    assert max(horizons) == 60
    assert max(horizons) <= 64


@pytest.mark.skipif(not GIFTEVAL_ROOT.is_dir(), reason="gift-eval checkout absent")
def test_gifteval_short_config_list_matches_the_checkout():
    """GIFTEVAL_SHORT_CONFIGS is transcribed from the benchmark's notebooks,
    so it is re-derived here to catch drift. Directory scanning would not do:
    the checkout also holds synthetic/* which is outside the benchmark."""
    notebook = json.loads((GIFTEVAL_ROOT / "notebooks/chronos-2.ipynb").read_text())
    source = "".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    canonical = re.search(r'SHORT_DATASETS\s*=\s*"([^"]+)"', source).group(1).split()

    assert list(suites.GIFTEVAL_SHORT_CONFIGS) == canonical
    assert len(canonical) == suites.EXPECTED_GIFTEVAL_CONFIGS
    absent = [c for c in canonical if not (GIFTEVAL_ROOT / c / "state.json").is_file()]
    assert absent == []


@pytest.mark.skipif(not MONASH_ROOT.is_dir(), reason="monash_eval absent")
@pytest.mark.parametrize(
    ("suite", "count", "horizons"),
    [
        ("m1", 1001, {6, 8, 18}),
        ("m3", 3003, {6, 8, 18}),
        ("tourism", 1311, {4, 8, 24}),
    ],
)
def test_monash_suites_load_the_canonical_series_counts(suite, count, horizons):
    """Cheap enough to run every time, unlike M4/Favorita/GIFT-Eval. The
    corpus copies of M1 and Tourism are truncated (921 and 1212), so this
    also guards against the loader being pointed back at them."""
    out = suites.load_suite(suite, {"monash": str(MONASH_ROOT)})
    assert len(out) == count
    assert {len(s.actual) for s in out} == horizons
    assert all(len(s.history) > 0 for s in out)

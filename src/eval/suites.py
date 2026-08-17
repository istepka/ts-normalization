"""Loaders for the six held-out evaluation suites.

Each loader returns `EvalSeries` records carrying the history a model may
condition on and the actuals it is scored against, following that suite's
own published protocol rather than any fixed context or horizon. The series
counts are asserted at load time, because a moved path or a re-download that
silently shrinks a suite would otherwise show up only as a slightly
different headline number.

Sources are fixed in notes/agentic_logs/2026-08-14-holdout-m1-m3.md. None of
them read from the held-out GiftEvalPretrain directories except Favorita,
which has no canonical release and whose corpus copy is the definition.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import datasets
import numpy as np
import pandas as pd

# Series counts verified against the canonical releases. A loader that finds
# a different count raises rather than scoring a truncated suite.
EXPECTED_SERIES = {
    "m1": 1001,
    "m3": 3003,
    "tourism": 1311,
    "m4": 100000,
    "favorita": 83207,
}
# GIFT-Eval's own SHORT_DATASETS list, transcribed from the benchmark's
# notebooks (e.g. notebooks/chronos-2.ipynb). This is the benchmark's
# definition of the short-term suite, so it is used verbatim rather than
# scanning directories: the checkout also holds `synthetic/*`, which is not
# part of the benchmark, and `jena_weather` appears both as a leaf and as
# frequency subdirectories. `test_gifteval_short_config_list_matches_checkout`
# re-derives this from the notebook and fails if it drifts.
GIFTEVAL_SHORT_CONFIGS = (
    "m4_yearly",
    "m4_quarterly",
    "m4_monthly",
    "m4_weekly",
    "m4_daily",
    "m4_hourly",
    "electricity/15T",
    "electricity/H",
    "electricity/D",
    "electricity/W",
    "solar/10T",
    "solar/H",
    "solar/D",
    "solar/W",
    "hospital",
    "covid_deaths",
    "us_births/D",
    "us_births/M",
    "us_births/W",
    "saugeenday/D",
    "saugeenday/M",
    "saugeenday/W",
    "temperature_rain_with_missing",
    "kdd_cup_2018_with_missing/H",
    "kdd_cup_2018_with_missing/D",
    "car_parts_with_missing",
    "restaurant",
    "hierarchical_sales/D",
    "hierarchical_sales/W",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "LOOP_SEATTLE/D",
    "SZ_TAXI/15T",
    "SZ_TAXI/H",
    "M_DENSE/H",
    "M_DENSE/D",
    "ett1/15T",
    "ett1/H",
    "ett1/D",
    "ett1/W",
    "ett2/15T",
    "ett2/H",
    "ett2/D",
    "ett2/W",
    "jena_weather/10T",
    "jena_weather/H",
    "jena_weather/D",
    "bitbrains_fast_storage/5T",
    "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T",
    "bitbrains_rnd/H",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
)
EXPECTED_GIFTEVAL_CONFIGS = 55

# M4 scores against its own competition seasonality so the standalone suite
# stays comparable to the M4 literature. It happens to agree with gluonts on
# all six subsets, but it is stated here rather than derived because that is
# a coincidence of the two conventions, not a guarantee.
M4_SPEC = {
    "Yearly": {"horizon": 6, "period": 1, "freq": "Y"},
    "Quarterly": {"horizon": 8, "period": 4, "freq": "Q"},
    "Monthly": {"horizon": 18, "period": 12, "freq": "M"},
    "Weekly": {"horizon": 13, "period": 1, "freq": "W"},
    "Daily": {"horizon": 14, "period": 1, "freq": "D"},
    "Hourly": {"horizon": 48, "period": 24, "freq": "H"},
}
EXPECTED_M4_COUNTS = {
    "Yearly": 23000,
    "Quarterly": 24000,
    "Monthly": 48000,
    "Weekly": 359,
    "Daily": 4227,
    "Hourly": 414,
}

# Monash frequency name -> MASE seasonal lag. M3 Other carries no frequency
# and no start timestamp in its .tsf, so it declares no seasonal cycle.
MONASH_PERIODS = {"yearly": 1, "quarterly": 4, "monthly": 12, None: 1}
MONASH_FREQ_ALIAS = {"yearly": "Y", "quarterly": "Q", "monthly": "M", None: "Y"}

# Transcribed from gift-eval's src/gift_eval/data.py. `test_gifteval_short`
# re-reads them from the checkout and fails if they drift, so this copy
# cannot go stale unnoticed. Only the SHORT term is evaluated, whose
# multiplier is 1, which is what keeps every horizon inside the models'
# native 128 steps and avoids autoregressive rollout entirely.
GIFTEVAL_M4_PRED_LENGTH = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
GIFTEVAL_PRED_LENGTH = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
GIFTEVAL_TEST_SPLIT = 0.1
GIFTEVAL_MAX_WINDOW = 20

FAVORITA_HORIZON = 16
# Daily retail with a strong weekly cycle. This is our own choice, Favorita
# having no canonical protocol; gluonts would use no cycle at all for daily.
FAVORITA_PERIOD = 7
FAVORITA_END = pd.Timestamp("2017-08-15")


@dataclass(frozen=True)
class EvalSeries:
    """One scored series. `history` is everything the model may condition on
    and `actual` is the horizon it is scored against, both already split by
    the suite's own protocol. `period` is the MASE seasonal lag."""

    suite: str
    subset: str
    item_id: str
    history: np.ndarray
    actual: np.ndarray
    period: int
    freq: str | None


def read_tsf(path: Path) -> tuple[dict, list[tuple[str, np.ndarray]]]:
    """Minimal Monash .tsf reader: returns the `@`-header and the series.

    Only what the eval loaders need is parsed. `@horizon` carries the
    competition horizon and `@frequency` the seasonal cycle, both of which
    the harness reads rather than hard-coding.
    """
    header, series = {}, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        in_data = False
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                key, _, value = line[1:].partition(" ")
                if key == "data":
                    in_data = True
                else:
                    header.setdefault(key, value)
                continue
            if not in_data:
                continue
            parts = line.split(":")
            name, values = parts[0], parts[-1]
            series.append(
                (
                    name,
                    np.array([float(v) for v in values.split(",")], dtype=np.float64),
                )
            )
    return header, series


def load_monash(root: Path, suite: str) -> list[EvalSeries]:
    """M1, M3, or Tourism from the canonical Monash .tsf releases.

    The competition protocol holds out the last `@horizon` points of every
    series, so history is whatever precedes them however short that is.
    """
    prefix = {"m1": "m1_", "m3": "m3_", "tourism": "tourism_"}[suite]
    out = []
    for directory in sorted(p for p in root.iterdir() if p.name.startswith(prefix)):
        header, series = read_tsf(next(directory.glob("*.tsf")))
        horizon = int(header["horizon"])
        freq_name = header.get("frequency")
        period = MONASH_PERIODS[freq_name]
        freq = MONASH_FREQ_ALIAS[freq_name]
        for name, values in series:
            if len(values) <= horizon:
                raise ValueError(
                    f"{directory.name}/{name} has {len(values)} points for a "
                    f"horizon of {horizon}, leaving no history"
                )
            out.append(
                EvalSeries(
                    suite=suite,
                    subset=directory.name,
                    item_id=name,
                    history=values[:-horizon],
                    actual=values[-horizon:],
                    period=period,
                    freq=freq,
                )
            )
    _check_count(suite, len(out))
    return out


def load_m4(root: Path) -> list[EvalSeries]:
    """M4 from the official competition CSVs, which ship the train/test split
    already made, so no splitting decision is taken here."""
    out = []
    for subset, spec in M4_SPEC.items():
        train = _read_m4_csv(root / "Train" / f"{subset}-train.csv")
        test = _read_m4_csv(root / "Test" / f"{subset}-test.csv")
        if len(train) != EXPECTED_M4_COUNTS[subset]:
            raise ValueError(
                f"M4 {subset}: {len(train)} series, expected "
                f"{EXPECTED_M4_COUNTS[subset]}"
            )
        for item_id, history in train.items():
            actual = test[item_id]
            if len(actual) != spec["horizon"]:
                raise ValueError(
                    f"M4 {item_id}: {len(actual)} test points, expected "
                    f"{spec['horizon']}"
                )
            out.append(
                EvalSeries(
                    suite="m4",
                    subset=subset,
                    item_id=item_id,
                    history=history,
                    actual=actual,
                    period=spec["period"],
                    freq=spec["freq"],
                )
            )
    _check_count("m4", len(out))
    return out


def _read_m4_csv(path: Path) -> dict[str, np.ndarray]:
    """M4 CSVs are ragged, every row padded with empty cells to the longest
    series, so trailing blanks are dropped rather than read as zeros."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            values = [float(v) for v in row[1:] if v != ""]
            out[row[0]] = np.array(values, dtype=np.float64)
    return out


def load_favorita(corpus_root: Path) -> list[EvalSeries]:
    """Favorita store-item daily sales, held out from pretraining.

    Protocol per notes/agentic_logs/2026-08-14-holdout-m1-m3.md: the last 16
    days are the horizon, every series reaching 2017-08-15 is eligible
    whatever its history length, and NaN means no sale so it is filled with
    zero per Kaggle's convention. No minimum context is imposed, which is the
    whole point of evaluating on the competition setup rather than on the
    training window shape.
    """
    dataset = datasets.load_from_disk(str(corpus_root / "favorita_sales"))
    out = []
    for batch in dataset.iter(batch_size=5000):
        for item_id, start, target in zip(
            batch["item_id"], batch["start"], batch["target"]
        ):
            values = np.asarray(target, dtype=np.float64)
            end = pd.Timestamp(start) + pd.Timedelta(days=len(values) - 1)
            if end != FAVORITA_END:
                continue
            values = np.nan_to_num(values, nan=0.0)
            out.append(
                EvalSeries(
                    suite="favorita",
                    subset="favorita_sales",
                    item_id=item_id,
                    history=values[:-FAVORITA_HORIZON],
                    actual=values[-FAVORITA_HORIZON:],
                    period=FAVORITA_PERIOD,
                    freq="D",
                )
            )
    _check_count("favorita", len(out))
    return out


def base_freq_code(freq: str) -> str:
    """The bare frequency letter behind a gluonts alias, e.g. "A-DEC" -> "A",
    "10T" -> "T", "W-SUN" -> "W". This is the key into GIFT-Eval's
    prediction-length maps."""
    base = freq.partition("-")[0]
    return base.lstrip("0123456789")


def load_gifteval_short(root: Path) -> list[EvalSeries]:
    """GIFT-Eval, short term only.

    Reimplements the short-term split from gift-eval's own `data.py` rather
    than importing it, which would pull in gluonts. The pieces reproduced are
    the prediction-length maps, the rolling-window count, and the
    non-overlapping test windows taken from the end of each series.
    Multivariate datasets are flattened to univariate first, as GIFT-Eval
    does via `MultivariateToUnivariate`.

    Only SHORT is evaluated, so the term multiplier is 1.
    """
    missing = [
        c for c in GIFTEVAL_SHORT_CONFIGS if not (root / c / "state.json").is_file()
    ]
    if missing:
        raise ValueError(f"GIFT-Eval configs absent from {root}: {missing}")

    out = []
    for name in GIFTEVAL_SHORT_CONFIGS:
        directory = root / name
        dataset = datasets.load_from_disk(str(directory))
        univariate = []
        for batch in dataset.iter(batch_size=64):
            for item_id, freq, target in zip(
                batch["item_id"], batch["freq"], batch["target"]
            ):
                target = np.asarray(target, dtype=np.float64)
                if target.ndim == 1:
                    univariate.append((str(item_id), freq, target))
                else:
                    for channel in range(target.shape[0]):
                        univariate.append(
                            (f"{item_id}_{channel}", freq, target[channel])
                        )

        code = base_freq_code(univariate[0][1])
        table = GIFTEVAL_M4_PRED_LENGTH if "m4" in name else GIFTEVAL_PRED_LENGTH
        if code not in table:
            raise ValueError(f"{name}: no short-term horizon for frequency {code!r}")
        horizon = table[code]

        if "m4" in name:
            windows = 1
        else:
            shortest = min(len(t) for _, _, t in univariate)
            windows = math.ceil(GIFTEVAL_TEST_SPLIT * shortest / horizon)
            windows = min(max(1, windows), GIFTEVAL_MAX_WINDOW)

        for item_id, freq, target in univariate:
            for window in range(windows, 0, -1):
                stop = len(target) - (window - 1) * horizon
                start = stop - horizon
                if start - 1 < 1:
                    raise ValueError(
                        f"{name}/{item_id} is too short for {windows} windows "
                        f"of {horizon}"
                    )
                out.append(
                    EvalSeries(
                        suite="gifteval",
                        subset=name,
                        item_id=f"{item_id}_w{window}",
                        history=target[:start],
                        actual=target[start:stop],
                        period=_gluonts_seasonality(freq),
                        freq=freq,
                    )
                )

    configs = {s.subset for s in out}
    if len(configs) != EXPECTED_GIFTEVAL_CONFIGS:
        raise ValueError(
            f"produced {len(configs)} GIFT-Eval configs, expected "
            f"{EXPECTED_GIFTEVAL_CONFIGS}"
        )
    return out


def load_suite(name: str, roots: dict[str, str]) -> list[EvalSeries]:
    """Loads one suite by name. `roots` maps the source keys in
    `conf/eval.yaml` to paths, and is indexed rather than `.get`-ed so a
    missing root fails here rather than at the first file read."""
    if name in ("m1", "m3", "tourism"):
        return load_monash(Path(roots["monash"]), name)
    if name == "m4":
        return load_m4(Path(roots["m4"]))
    if name == "gifteval":
        return load_gifteval_short(Path(roots["gifteval"]))
    if name == "favorita":
        return load_favorita(Path(roots["corpus"]))
    raise ValueError(f"unknown suite {name!r}, expected one of {SUITES}")


SUITES = ("m1", "m3", "tourism", "m4", "gifteval", "favorita")


# gluonts.time_feature.seasonality.DEFAULT_SEASONALITIES, which is what
# GIFT-Eval's evaluator uses. Deliberately NOT src/data/seasonality.py: that
# module maps daily to the weekly cycle (7) for the training corpus, whereas
# gluonts uses 1, and it maps seconds to the daily cycle where gluonts uses
# the hourly one. Scoring GIFT-Eval on the corpus convention put our
# seasonal-naive MASE 1.5x to 5x away from the benchmark's published numbers
# on every daily and 10-second config.
GLUONTS_SEASONALITIES = {
    "S": 3600,
    "T": 1440,
    "H": 24,
    "D": 1,
    "W": 1,
    "M": 12,
    "B": 5,
    "Q": 4,
    "A": 1,
    "Y": 1,
}


def _gluonts_seasonality(freq: str) -> int:
    """GIFT-Eval scores on the gluonts seasonality convention, so this is
    deliberately separate from the M4 competition periods above.

    Mirrors `gluonts.time_feature.get_seasonality`: divide the base
    seasonality by the frequency's multiplier, falling back to 1 when it does
    not divide evenly.
    """
    base = GLUONTS_SEASONALITIES.get(base_freq_code(freq), 1)
    multiplier = _freq_multiplier(freq)
    seasonality, remainder = divmod(base, multiplier)
    return seasonality if remainder == 0 else 1


def _freq_multiplier(freq: str) -> int:
    digits = ""
    for char in freq.partition("-")[0]:
        if not char.isdigit():
            break
        digits += char
    return int(digits) if digits else 1


def _check_count(suite: str, found: int) -> None:
    expected = EXPECTED_SERIES[suite]
    if found != expected:
        raise ValueError(
            f"{suite}: loaded {found} series, expected {expected}. The suite "
            "definition is fixed in notes/agentic_logs/2026-08-14-holdout-m1-m3.md; "
            "a mismatch means the source moved or was re-downloaded, not that "
            "the expectation should be edited."
        )

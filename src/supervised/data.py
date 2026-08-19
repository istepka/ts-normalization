"""Canonical M-series and Tourism data splits for supervised forecasting."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.seasonality import parse_offset
from src.eval import suites as eval_suites


@dataclass(frozen=True)
class SupervisedSeries:
    """One complete canonical Monash series before supervised splitting."""

    suite: str
    subset: str
    item_id: str
    values: np.ndarray
    period: int
    freq: str
    official_horizon: int

    @property
    def unique_id(self) -> str:
        return f"{self.suite}/{self.subset}/{self.item_id}"


@dataclass(frozen=True)
class SupervisedSplit:
    """Leakage-free train, validation, and rolling-test regions."""

    item: SupervisedSeries
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    validation_start: int
    test_start: int


def load_series(
    root: Path,
    suite_names: tuple[str, ...] = ("m1", "tourism"),
    m4_root: Path | None = None,
):
    """Loads complete canonical M1, M3, M4, and Tourism series."""
    out = []
    for suite_name in suite_names:
        if suite_name not in ("m1", "m3", "m4", "tourism"):
            raise ValueError(f"unsupported supervised suite {suite_name!r}")
        if suite_name == "m4":
            if m4_root is None:
                raise ValueError("m4_root is required when suite m4 is selected")
            source_series = eval_suites.load_m4(m4_root)
        else:
            source_series = eval_suites.load_monash(root, suite_name)
        for item in source_series:
            values = np.concatenate((item.history, item.actual)).astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{item.suite}/{item.item_id} contains non-finite values"
                )
            if item.freq is None:
                raise ValueError(f"{item.suite}/{item.item_id} has no frequency")
            out.append(
                SupervisedSeries(
                    suite=item.suite,
                    subset=item.subset,
                    item_id=item.item_id,
                    values=values,
                    period=item.period,
                    freq=item.freq,
                    official_horizon=len(item.actual),
                )
            )
    return out


def eligible_series(
    series: list[SupervisedSeries],
    validation_size: int,
    minimum_train_size: int,
) -> list[SupervisedSeries]:
    """Keeps series that can provide one complete supervised training window."""
    if validation_size < 1 or minimum_train_size < 1:
        raise ValueError("validation_size and minimum_train_size must be positive")
    out = []
    for item in series:
        test_size = 2 * item.official_horizon - 1
        validation_start = len(item.values) - test_size - validation_size
        if validation_start >= minimum_train_size:
            out.append(item)
    return out


def split_series(
    series: list[SupervisedSeries], validation_size: int
) -> list[SupervisedSplit]:
    """Reserves validation and a `2H-1` test tail for every series.

    The test length uses each series' official horizon. `validation_size` is
    the common model horizon for a pooled frequency run.
    """
    if validation_size < 1:
        raise ValueError("validation_size must be positive")
    out = []
    for item in series:
        test_size = 2 * item.official_horizon - 1
        test_start = len(item.values) - test_size
        validation_start = test_start - validation_size
        if validation_start < 1:
            raise ValueError(
                f"{item.unique_id} has {len(item.values)} points, but needs "
                f"{validation_size + test_size + 1} for the requested split"
            )
        out.append(
            SupervisedSplit(
                item=item,
                train=item.values[:validation_start],
                validation=item.values[validation_start:test_start],
                test=item.values[test_start:],
                validation_start=validation_start,
                test_start=test_start,
            )
        )
    return out


def frequency_groups(
    series: list[SupervisedSeries],
) -> dict[str, list[SupervisedSeries]]:
    """Groups series by frequency."""
    grouped: dict[str, list[SupervisedSeries]] = {}
    for item in series:
        grouped.setdefault(item.freq, []).append(item)
    return dict(sorted(grouped.items()))


def model_horizon(series: list[SupervisedSeries]) -> int:
    """Common output horizon for one pooled frequency experiment."""
    return max(item.official_horizon for item in series)


def context_length(series: list[SupervisedSeries]) -> int:
    """Two forecast horizons or two seasonal cycles, whichever is longer."""
    horizon = model_horizon(series)
    period = max(item.period for item in series)
    return max(2 * horizon, 2 * period)


def _frame_rows(
    values_by_item: list[tuple[SupervisedSeries, np.ndarray]], freq: str
) -> list[dict]:
    rows = []
    for item, values in values_by_item:
        dates = pd.date_range(
            "2000-01-01", periods=len(values), freq=parse_offset(freq)
        )
        rows.extend(
            {
                "unique_id": item.unique_id,
                "ds": date,
                "y": float(value),
            }
            for date, value in zip(dates, values)
        )
    return rows


def training_frame(splits: list[SupervisedSplit], freq: str) -> pd.DataFrame:
    """Returns train plus validation values for NeuralForecast.fit.

    NeuralForecast's `val_size` then takes the final common model horizon as
    validation, leaving the train region untouched.
    """
    values = [
        (split.item, np.concatenate((split.train, split.validation)))
        for split in splits
    ]
    return pd.DataFrame.from_records(_frame_rows(values, freq))


def history_frame(
    items: list[SupervisedSeries], histories: list[np.ndarray], freq: str
) -> pd.DataFrame:
    """Builds a prediction frame from one history per series."""
    if len(items) != len(histories):
        raise ValueError(f"{len(items)} items for {len(histories)} histories")
    return pd.DataFrame.from_records(_frame_rows(list(zip(items, histories)), freq))

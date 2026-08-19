"""Accuracy evaluation for supervised NeuralForecast checkpoints."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast

from src.eval import predict, score
from src.eval.suites import EvalSeries
from src.metrics import accuracy
from src.supervised.causal import CausalForecaster
from src.supervised.data import SupervisedSeries, SupervisedSplit, history_frame


def _eval_item(
    item: SupervisedSeries, history: np.ndarray, actual: np.ndarray
) -> EvalSeries:
    return EvalSeries(
        suite=item.suite,
        subset=item.subset,
        item_id=item.item_id,
        history=history,
        actual=actual,
        period=item.period,
        freq=item.freq,
    )


def _prediction_column(predictions: pd.DataFrame) -> str:
    columns = [
        column for column in predictions.columns if column not in ("unique_id", "ds")
    ]
    if len(columns) != 1:
        raise ValueError(f"expected one NeuralForecast output column, got {columns}")
    return columns[0]


def _predict(
    forecaster: NeuralForecast | CausalForecaster,
    items: list[SupervisedSeries],
    histories: list[np.ndarray],
    freq: str,
    horizon: int,
) -> dict[str, np.ndarray]:
    if isinstance(forecaster, CausalForecaster):
        return forecaster.predict(items, histories, freq, horizon)
    frame = history_frame(items, histories, freq)
    predictions = forecaster.predict(frame, h=horizon, verbose=False)
    column = _prediction_column(predictions)
    grouped = {
        unique_id: group[column].to_numpy(dtype=np.float64)
        for unique_id, group in predictions.groupby("unique_id", sort=False)
    }
    if set(grouped) != {item.unique_id for item in items}:
        raise ValueError("NeuralForecast did not return one forecast per input series")
    return grouped


def _score_group(
    model_name: str,
    items: list[SupervisedSeries],
    histories: list[np.ndarray],
    actuals: list[np.ndarray],
    predictions: dict[str, np.ndarray],
    context_length: int,
    phase: str,
    origin_offset: int,
) -> list[dict]:
    rows = []
    groups = sorted({(item.suite, len(actual)) for item, actual in zip(items, actuals)})
    for suite, official_horizon in groups:
        indexes = [
            i
            for i, (item, actual) in enumerate(zip(items, actuals))
            if item.suite == suite and len(actual) == official_horizon
        ]
        grouped_items = [items[i] for i in indexes]
        grouped_histories = [histories[i] for i in indexes]
        grouped_actuals = [actuals[i] for i in indexes]
        eval_items = [
            _eval_item(item, history, actual)
            for item, history, actual in zip(
                grouped_items, grouped_histories, grouped_actuals
            )
        ]
        context, context_mask = predict.build_context(eval_items, context_length)
        values = np.stack(
            [predictions[item.unique_id][:official_horizon] for item in grouped_items]
        )[:, :, None]
        forecasts = predict.Forecasts(
            values=values,
            actual=np.stack(grouped_actuals),
            actual_mask=np.ones((len(eval_items), official_horizon)),
            history=context,
            history_mask=context_mask,
            periods=np.array([item.period for item in eval_items]),
            quantiles=[0.5],
            subsets=[item.subset for item in eval_items],
            item_ids=[item.item_id for item in eval_items],
        )
        model_scores = score.score(forecasts, eval_items)
        model_metrics = accuracy.pool(model_scores)
        baseline = score.seasonal_naive(eval_items, official_horizon, 1)
        baseline_forecasts = predict.Forecasts(
            values=baseline,
            actual=forecasts.actual,
            actual_mask=forecasts.actual_mask,
            history=forecasts.history,
            history_mask=forecasts.history_mask,
            periods=forecasts.periods,
            quantiles=[0.5],
            subsets=forecasts.subsets,
            item_ids=forecasts.item_ids,
        )
        baseline_scores = score.score(baseline_forecasts, eval_items)
        baseline_metrics = accuracy.pool(baseline_scores)
        row = {
            "model": model_name,
            "phase": phase,
            "origin_offset": origin_offset,
            "subset": suite,
            "freq": eval_items[0].freq,
            "horizon": official_horizon,
            "n_series": len(eval_items),
        }
        row.update({f"model_{key}": value for key, value in model_metrics.items()})
        model_wql_usable = (
            np.isfinite(model_scores["wql_num"])
            & np.isfinite(model_scores["wql_den"])
            & (model_scores["wql_den"] > 0)
        )
        row["model_wql_num"] = float(model_scores["wql_num"][model_wql_usable].sum())
        row["model_wql_den"] = float(model_scores["wql_den"][model_wql_usable].sum())
        row.update(
            {f"seasonal_naive_{key}": value for key, value in baseline_metrics.items()}
        )
        baseline_wql_usable = (
            np.isfinite(baseline_scores["wql_num"])
            & np.isfinite(baseline_scores["wql_den"])
            & (baseline_scores["wql_den"] > 0)
        )
        row["seasonal_naive_wql_num"] = float(
            baseline_scores["wql_num"][baseline_wql_usable].sum()
        )
        row["seasonal_naive_wql_den"] = float(
            baseline_scores["wql_den"][baseline_wql_usable].sum()
        )
        rows.append(row)
    return rows


def evaluate_validation(
    forecaster: NeuralForecast | CausalForecaster,
    splits: list[SupervisedSplit],
    freq: str,
    model_horizon: int,
    context_length: int,
    model_name: str,
) -> list[dict]:
    """Scores the first official horizon from the held-out validation tail."""
    items = [split.item for split in splits]
    histories = [split.item.values[: split.validation_start] for split in splits]
    actuals = [split.validation[: split.item.official_horizon] for split in splits]
    predictions = _predict(forecaster, items, histories, freq, model_horizon)
    return _score_group(
        model_name,
        items,
        histories,
        actuals,
        predictions,
        context_length,
        "validation",
        0,
    )


def evaluate_test_origins(
    forecaster: NeuralForecast | CausalForecaster,
    splits: list[SupervisedSplit],
    freq: str,
    model_horizon: int,
    context_length: int,
    model_name: str,
) -> list[dict]:
    """Scores every H-step rolling origin in each series' `2H-1` test tail."""
    rows = []
    max_offset = max(split.item.official_horizon for split in splits)
    for origin_offset in range(max_offset):
        active = [
            split for split in splits if origin_offset < split.item.official_horizon
        ]
        items = [split.item for split in active]
        histories = [
            split.item.values[: split.test_start + origin_offset] for split in active
        ]
        actuals = [
            split.item.values[
                split.test_start + origin_offset : split.test_start
                + origin_offset
                + split.item.official_horizon
            ]
            for split in active
        ]
        predictions = _predict(forecaster, items, histories, freq, model_horizon)
        rows.extend(
            _score_group(
                model_name,
                items,
                histories,
                actuals,
                predictions,
                context_length,
                "test",
                origin_offset,
            )
        )
    return rows


def aggregate_test_origins(rows: Sequence[dict]) -> dict[str, float]:
    """Pools rolling-origin rows weighted by the number of scored series."""
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("cannot aggregate empty rolling-origin results")
    ratio_metrics = {
        column.removesuffix("_num")
        for column in table.columns
        if column.endswith("_num")
        and f"{column.removesuffix('_num')}_den" in table.columns
    }
    ratio_halves = {
        f"{metric}_{suffix}" for metric in ratio_metrics for suffix in ("num", "den")
    }
    metrics = [
        column
        for column in table.columns
        if column.startswith(("model_", "seasonal_naive_"))
        and not column.endswith("_n")
        and column not in ratio_metrics
        and column not in ratio_halves
    ]
    aggregated = {}
    for metric in metrics:
        values = table[metric].to_numpy(dtype=np.float64)
        counts = table[f"{metric}_n"].to_numpy(dtype=np.float64)
        usable = np.isfinite(values) & (counts > 0)
        aggregated[metric] = (
            float(np.average(values[usable], weights=counts[usable]))
            if usable.any()
            else float("nan")
        )
        aggregated[f"{metric}_n"] = int(counts[usable].sum())
    for metric in sorted(ratio_metrics):
        numerator = table[f"{metric}_num"].to_numpy(dtype=np.float64).sum()
        denominator = table[f"{metric}_den"].to_numpy(dtype=np.float64).sum()
        aggregated[metric] = (
            float(numerator / denominator) if denominator > 0 else float("nan")
        )
        aggregated[f"{metric}_n"] = int(table[f"{metric}_n"].sum())
    return aggregated

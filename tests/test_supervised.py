from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.plotting.scripts.generate_supervised_mseries_tables import (
    aggregate_metric,
    paired_effect,
)
from src.supervised.causal import _prefix_statistics, causal_training_windows
from src.supervised.data import (
    SupervisedSeries,
    context_length,
    eligible_series,
    model_horizon,
    split_series,
    training_frame,
)
from src.supervised.evaluate import aggregate_test_origins
from src.supervised.models import (
    CONDITIONS,
    MODEL_CLASSES,
    BestValidationWeights,
    build_model,
)


def _series(name, horizon, period, length):
    return SupervisedSeries(
        suite="m1" if name == "m1" else "tourism",
        subset=f"{name}_monthly",
        item_id="T1",
        values=np.arange(length, dtype=np.float64),
        period=period,
        freq="M",
        official_horizon=horizon,
    )


def test_eligibility_requires_one_complete_training_window():
    series = [_series("m1", horizon=4, period=4, length=18)]

    assert eligible_series(series, validation_size=4, minimum_train_size=8) == []
    assert eligible_series(series, validation_size=4, minimum_train_size=4) == series


def test_supervised_split_reserves_validation_and_two_horizon_test_tail():
    series = [_series("m1", horizon=4, period=12, length=30)]
    split = split_series(series, validation_size=6)[0]

    assert len(split.train) == 17
    assert len(split.validation) == 6
    assert len(split.test) == 7
    assert np.array_equal(split.test, np.arange(23, 30, dtype=np.float64))
    assert np.array_equal(split.train[-1:], [16.0])


def test_context_uses_common_horizon_and_seasonal_cycle():
    series = [_series("m1", 18, 12, 80), _series("tourism", 24, 12, 80)]

    assert model_horizon(series) == 24
    assert context_length(series) == 48


def test_supervised_config_has_bounded_early_stopping_budget():
    cfg = OmegaConf.load("conf/supervised.yaml")

    assert cfg.train.max_steps == 20_000
    assert cfg.train.val_check_steps == 250
    assert cfg.train.early_stop_patience_steps == 8
    assert "supervised_early_stop" in cfg.output_dir


def test_best_validation_callback_restores_best_weights():
    callback = BestValidationWeights()
    model = torch.nn.Linear(1, 1, bias=False)
    trainer = SimpleNamespace(callback_metrics={"ptl/val_loss": torch.tensor(1.0)})
    model.weight.data.fill_(2.0)
    callback.on_validation_end(trainer, model)
    trainer.callback_metrics["ptl/val_loss"] = torch.tensor(2.0)
    model.weight.data.fill_(3.0)
    callback.on_validation_end(trainer, model)

    callback.on_train_end(trainer, model)

    assert model.weight.item() == 2.0


def test_training_frame_keeps_validation_at_the_end_of_each_series():
    series = [_series("m1", 4, 12, 30)]
    split = split_series(series, validation_size=6)
    frame = training_frame(split, "M")

    assert list(frame.columns) == ["unique_id", "ds", "y"]
    assert len(frame) == 23
    assert isinstance(frame["ds"].iloc[0], pd.Timestamp)
    assert frame["y"].iloc[-1] == 22.0


def test_causal_statistics_use_all_values_through_context_only():
    values = np.array([1.0, 2.0, 3.0, 100.0])

    location, scale = _prefix_statistics(values, np.array([3]))

    assert location[0] == 2.0
    assert np.isclose(scale[0], np.sqrt(2.0 / 3.0) + 1e-6)


def test_causal_windows_exclude_future_from_statistics():
    series = [_series("m1", 4, 4, 30)]
    split = split_series(series, validation_size=4)[0]
    windows = causal_training_windows([split], input_size=4, horizon=2)

    assert np.array_equal(windows.contexts[0].numpy(), [0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(windows.targets[0].numpy(), [4.0, 5.0])
    assert windows.locations[0].item() == 1.5


def test_all_neuralforecast_models_build_both_supervised_conditions():
    for name in MODEL_CLASSES:
        for condition in CONDITIONS:
            model = build_model(
                name=name,
                condition=condition,
                horizon=4,
                input_size=8,
                max_steps=1,
                batch_size=2,
                windows_batch_size=2,
                learning_rate=1.0e-3,
                val_check_steps=1,
                early_stop_patience_steps=-1,
                num_lr_decays=0,
                seed=0,
                device="cpu",
            )
            assert model.h == 4
            assert model.input_size == 8
            assert not model.start_padding_enabled


def test_rolling_origin_wql_aggregates_as_ratio_of_sums():
    rows = [
        {
            "n_series": 1,
            "model_mae": 2.0,
            "model_mae_n": 1,
            "model_wql": 1.0,
            "model_wql_n": 1,
            "model_wql_num": 2.0,
            "model_wql_den": 2.0,
        },
        {
            "n_series": 1,
            "model_mae": 20.0,
            "model_mae_n": 1,
            "model_wql": 0.1,
            "model_wql_n": 1,
            "model_wql_num": 20.0,
            "model_wql_den": 200.0,
        },
    ]

    aggregated = aggregate_test_origins(rows)

    assert aggregated["model_mae"] == 11.0
    assert np.isclose(aggregated["model_wql"], 22.0 / 202.0)
    assert "model_wql_num" not in aggregated
    assert "model_wql_den" not in aggregated


def test_rolling_origin_metrics_use_their_own_valid_counts():
    rows = [
        {"n_series": 10, "model_mape": 20.0, "model_mape_n": 2},
        {"n_series": 10, "model_mape": 50.0, "model_mape_n": 8},
    ]

    aggregated = aggregate_test_origins(rows)

    assert aggregated["model_mape"] == 44.0
    assert aggregated["model_mape_n"] == 10


def test_supervised_table_effect_is_paired_across_frequencies():
    results = {}
    for frequency in ("Y", "Q", "M", "W", "D", "H"):
        results[("nhits", frequency, "sit", "standard")] = {
            "test_aggregate": {"model_wql": 2.0}
        }
        results[("nhits", frequency, "sit", "causal")] = {
            "test_aggregate": {"model_wql": 1.0}
        }

    change, wins = paired_effect(
        results,
        model="nhits",
        metric="model_wql",
        comparison="causal_standard",
        fixed_condition="sit",
    )

    assert np.isclose(change, -50.0)
    assert wins == 6


def test_supervised_table_mase_uses_valid_series_counts():
    rows = [
        {"model_mase": 1.0, "model_mase_n": 2},
        {"model_mase": 4.0, "model_mase_n": 8},
    ]

    assert aggregate_metric(rows, "mase", "model") == 3.4

"""Re-evaluate supervised checkpoints with benchmark-level metrics."""

import argparse
import json
from pathlib import Path

import torch
from neuralforecast import NeuralForecast
from omegaconf import OmegaConf

from src.models.normalization import PopulationStdScheme
from src.supervised.causal import CausalForecaster
from src.supervised.data import (
    eligible_frequency_series,
    load_series,
    split_series,
)
from src.supervised.evaluate import (
    aggregate_test_origins,
    evaluate_test_origins,
    evaluate_validation,
)
from src.supervised.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / "resolved_config.yaml")
    root = Path(cfg.data.monash_root)
    m4_root = Path(cfg.data.m4_root) if "m4" in cfg.data.suites else None
    series = load_series(root, tuple(cfg.data.suites), m4_root)
    _, frequency_series, horizon, input_size = eligible_frequency_series(
        series, cfg.frequency
    )
    splits = split_series(frequency_series, horizon)

    if cfg.normalization == "standard":
        forecaster = NeuralForecast.load(
            str(args.run_dir / "neuralforecast"),
            map_location=torch.device(args.device),
        )
        for model in forecaster.models:
            model.trainer_kwargs["accelerator"] = (
                "gpu" if args.device == "cuda" else "cpu"
            )
            model.trainer_kwargs["devices"] = 1
    elif cfg.normalization == "causal":
        model = build_model(
            name=cfg.model,
            condition=cfg.condition,
            horizon=horizon,
            input_size=input_size,
            max_steps=cfg.train.max_steps,
            batch_size=cfg.train.batch_size,
            windows_batch_size=cfg.train.windows_batch_size,
            learning_rate=cfg.train.learning_rate,
            val_check_steps=cfg.train.val_check_steps,
            early_stop_patience_steps=cfg.train.early_stop_patience_steps,
            num_lr_decays=cfg.train.num_lr_decays,
            seed=cfg.seed,
            device=args.device,
            normalization_mode=cfg.normalization,
        )
        checkpoint = torch.load(
            args.run_dir / "causal_model.pt",
            map_location=args.device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["state_dict"])
        forecaster = CausalForecaster(
            model=model.to(args.device),
            condition=checkpoint["condition"],
            input_size=checkpoint["input_size"],
            horizon=checkpoint["horizon"],
            device=args.device,
            scheme=PopulationStdScheme(eps=1e-6),
        )
    else:
        raise ValueError(f"unsupported scaling statistics {cfg.normalization!r}")

    validation = evaluate_validation(
        forecaster,
        splits,
        cfg.frequency,
        horizon,
        input_size,
        cfg.model,
    )
    test_origins = evaluate_test_origins(
        forecaster,
        splits,
        cfg.frequency,
        horizon,
        input_size,
        cfg.model,
    )
    benchmarks = sorted({row["subset"] for row in test_origins})
    output = {
        "model": cfg.model,
        "condition": cfg.condition,
        "normalization": cfg.normalization,
        "frequency": cfg.frequency,
        "validation": validation,
        "test_origins": test_origins,
        "test_aggregate": aggregate_test_origins(test_origins),
        "test_by_benchmark": {
            benchmark: aggregate_test_origins(
                [row for row in test_origins if row["subset"] == benchmark]
            )
            for benchmark in benchmarks
        },
    }
    (args.run_dir / "metrics_by_benchmark.json").write_text(
        json.dumps(output, indent=2, allow_nan=True)
    )


if __name__ == "__main__":
    main()

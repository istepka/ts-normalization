"""Hydra entry point for supervised M1 and Tourism training."""

import json
from pathlib import Path

import hydra
import wandb
from neuralforecast import NeuralForecast
from omegaconf import DictConfig, OmegaConf

from src.config.base import validate_config
from src.config.supervised import SupervisedConfig
from src.data.seasonality import parse_offset
from src.supervised.causal import train_causal
from src.supervised.data import (
    context_length,
    eligible_series,
    frequency_groups,
    load_series,
    model_horizon,
    split_series,
    training_frame,
)
from src.supervised.evaluate import (
    aggregate_test_origins,
    evaluate_test_origins,
    evaluate_validation,
)
from src.supervised.models import build_model


@hydra.main(version_base=None, config_path="../../conf", config_name="supervised")
def main(cfg: DictConfig) -> None:
    validate_config(cfg, SupervisedConfig)
    if cfg.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is unavailable")
    root = Path(cfg.data.monash_root)
    m4_root = Path(cfg.data.m4_root) if "m4" in cfg.data.suites else None
    series = load_series(root, tuple(cfg.data.suites), m4_root)
    groups = frequency_groups(series)
    if cfg.frequency not in groups:
        raise ValueError(
            f"frequency {cfg.frequency!r} is absent; available {tuple(groups)}"
        )
    source_series = groups[cfg.frequency]
    horizon = model_horizon(source_series)
    input_size = context_length(source_series)
    frequency_series = eligible_series(source_series, horizon, input_size + horizon)
    if not frequency_series:
        raise ValueError(
            f"frequency {cfg.frequency!r} has no series with a complete "
            "supervised training window"
        )
    eligible_ids = {item.unique_id for item in frequency_series}
    excluded_series = [
        item.unique_id for item in source_series if item.unique_id not in eligible_ids
    ]
    splits = split_series(frequency_series, horizon)
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
        device=cfg.device,
        normalization_mode=cfg.normalization,
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    (output_dir / "split_meta.json").write_text(
        json.dumps(
            {
                "frequency": cfg.frequency,
                "normalization": cfg.normalization,
                "model_horizon": horizon,
                "context_length": input_size,
                "suites": list(cfg.data.suites),
                "n_source_series": len(source_series),
                "n_series": len(splits),
                "excluded_series": excluded_series,
                "n_series_by_suite": {
                    suite: sum(split.item.suite == suite for split in splits)
                    for suite in cfg.data.suites
                },
                "official_horizons": sorted(
                    {split.item.official_horizon for split in splits}
                ),
            },
            indent=2,
        )
    )
    wandb_run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=(
            f"{cfg.model}-{cfg.condition}-{cfg.normalization}-"
            f"{cfg.frequency}-seed{cfg.seed}"
        ),
        group=f"{cfg.wandb.experiment}/{cfg.model}/{cfg.frequency}",
        job_type=f"{cfg.condition}-{cfg.normalization}",
        tags=[cfg.model, cfg.condition, cfg.normalization, cfg.frequency],
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    try:
        if cfg.normalization == "standard":
            nf_frequency = parse_offset(cfg.frequency).freqstr
            forecaster = NeuralForecast(models=[model], freq=nf_frequency)
            forecaster.fit(
                df=training_frame(splits, cfg.frequency),
                val_size=horizon,
                verbose=cfg.train.verbose,
            )
            model_dir = output_dir / "neuralforecast"
            forecaster.save(str(model_dir), save_dataset=False, overwrite=True)
        else:
            forecaster = train_causal(
                model=model,
                splits=splits,
                input_size=input_size,
                horizon=horizon,
                condition=cfg.condition,
                max_steps=cfg.train.max_steps,
                batch_size=cfg.train.batch_size,
                windows_batch_size=cfg.train.windows_batch_size,
                learning_rate=cfg.train.learning_rate,
                val_check_steps=cfg.train.val_check_steps,
                early_stop_patience_steps=cfg.train.early_stop_patience_steps,
                num_lr_decays=cfg.train.num_lr_decays,
                seed=cfg.seed,
                device=cfg.device,
            )
            forecaster.save(output_dir / "causal_model.pt")
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
        summary = {
            "model": cfg.model,
            "condition": cfg.condition,
            "normalization": cfg.normalization,
            "frequency": cfg.frequency,
            "model_horizon": horizon,
            "context_length": input_size,
            "validation": validation,
            "test_origins": test_origins,
            "test_aggregate": aggregate_test_origins(test_origins),
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True)
        )
        wandb_run.log(summary["test_aggregate"])
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    main()

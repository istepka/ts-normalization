"""Structured configuration for supervised M-series forecasting runs."""

from dataclasses import dataclass, field

from omegaconf import MISSING

from src.config.base import WandbConfig


@dataclass
class SupervisedDataConfig:
    monash_root: str = MISSING
    m4_root: str = MISSING
    suites: list[str] = field(default_factory=lambda: ["m1", "m3", "m4", "tourism"])


@dataclass
class SupervisedTrainConfig:
    max_steps: int = MISSING
    batch_size: int = MISSING
    windows_batch_size: int = MISSING
    learning_rate: float = MISSING
    val_check_steps: int = MISSING
    early_stop_patience_steps: int = MISSING
    num_lr_decays: int = MISSING
    verbose: bool = MISSING


@dataclass
class SupervisedConfig:
    model: str = MISSING
    condition: str = MISSING
    normalization: str = MISSING
    frequency: str = MISSING
    device: str = MISSING
    seed: int = MISSING
    output_dir: str = MISSING
    data: SupervisedDataConfig = field(default_factory=SupervisedDataConfig)
    train: SupervisedTrainConfig = field(default_factory=SupervisedTrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

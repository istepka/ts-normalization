"""Structured schema for the synthetic loss-space toy experiment."""

from dataclasses import dataclass, field

from omegaconf import MISSING

from src.config.base import WandbConfig


@dataclass
class ToyCategoryConfig:
    name: str = MISSING
    freq: float = MISSING
    phase: float = MISSING
    scale: float = MISSING


@dataclass
class ToySourceConfig:
    name: str = MISSING
    path: str = MISSING
    key: str = MISSING


@dataclass
class ToyDataConfig:
    kind: str = MISSING
    real_shape_path: str = MISSING
    real_shape_key: str = MISSING
    real_shape_val_fraction: float = MISSING
    real_value_scale: float = MISSING
    real_source_root: str = MISSING
    real_sources: list[ToySourceConfig] = field(default_factory=list)
    scale_assignment: list[float] = field(default_factory=list)
    mean: float = MISSING
    categories: list[ToyCategoryConfig] = field(default_factory=list)
    cycle_length: int = MISSING
    series_length: int = MISSING
    series_per_category: int = MISSING
    val_series_per_category: int = MISSING
    val_windows_per_category: int = MISSING
    context_length: int = MISSING
    horizon: int = MISSING
    equal_variance: bool = MISSING


@dataclass
class ToyModelConfig:
    patch_length: int = MISSING
    d_model: int = MISSING
    n_heads: int = MISSING
    n_layers: int = MISSING
    dim_feedforward: int = MISSING
    dropout: float = MISSING
    norm_eps: float = MISSING


@dataclass
class ScheduleConfig:
    until: int = MISSING
    every: int = MISSING


@dataclass
class ToyTrainConfig:
    steps: int = MISSING
    batch_size: int = MISSING
    optimizer: str = MISSING
    lr: float = MISSING
    lr_adjusted: float = MISSING
    eval_schedule: list[ScheduleConfig] = field(default_factory=list)
    grad_norm_match: bool = MISSING
    forecast_schedule: list[ScheduleConfig] = field(default_factory=list)
    forecast_columns: list[int] = field(default_factory=list)


@dataclass
class ToyPlotConfig:
    render: bool = MISSING
    band: str = MISSING
    linear_xlim: list[int] = field(default_factory=list)


@dataclass
class ToyConfig:
    seeds: list[int] = field(default_factory=list)
    device: str = MISSING
    output_dir: str = MISSING
    setups: list[str] = field(default_factory=list)
    data: ToyDataConfig = field(default_factory=ToyDataConfig)
    model: ToyModelConfig = field(default_factory=ToyModelConfig)
    train: ToyTrainConfig = field(default_factory=ToyTrainConfig)
    plot: ToyPlotConfig = field(default_factory=ToyPlotConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

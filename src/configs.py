"""Structured schemas for the Hydra experiment configurations."""

from dataclasses import dataclass, field

from omegaconf import MISSING, DictConfig, OmegaConf


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
class WandbConfig:
    entity: str = MISSING
    project: str = MISSING
    mode: str = MISSING
    experiment: str = MISSING


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


@dataclass
class CorpusConfig:
    root: str = MISSING
    datasets: list[str] | None = None


@dataclass
class WindowIndexConfig:
    context_length: int = MISSING
    prediction_length: int = MISSING
    stride: int = MISSING
    val_series_fraction: float = MISSING
    min_valid_fraction: float = MISSING
    base_seed: int = MISSING
    max_windows_per_series: int | None = None
    cache_path: str | None = None


@dataclass
class TsfmTrainConfig:
    steps: int = MISSING
    batch_size: int = MISSING
    lr: float = MISSING
    optimizer: str = MISSING
    deterministic: bool = MISSING
    schedule_seed: int = MISSING
    eval_every: int = MISSING
    eval_batches: int = MISSING
    eval_windows_per_dataset: int = MISSING
    checkpoint_every: int = MISSING


@dataclass
class Chronos2Schema:
    context_length: int = MISSING
    prediction_length: int = MISSING
    patch_size: int = MISSING
    d_model: int = MISSING
    d_kv: int = MISSING
    d_ff: int = MISSING
    num_layers: int = MISSING
    num_heads: int = MISSING
    dropout_rate: float = MISSING
    initializer_factor: float = MISSING
    quantiles: list[float] = field(default_factory=list)
    use_arcsinh: bool = MISSING
    grad_clip_norm: float = MISSING


@dataclass
class Moirai2Schema:
    context_length: int = MISSING
    predict_horizon: int = MISSING
    patch_size: int = MISSING
    d_model: int = MISSING
    d_ff: int = MISSING
    num_layers: int = MISSING
    max_seq_len: int = MISSING
    attn_dropout_p: float = MISSING
    dropout_p: float = MISSING
    scaling: bool = MISSING
    quantile_levels: list[float] = field(default_factory=list)
    grad_clip_norm: float = MISSING


@dataclass
class MomentSchema:
    context_length: int = MISSING
    patch_len: int = MISSING
    d_model: int = MISSING
    t5_num_layers: int = MISSING
    t5_num_heads: int = MISSING
    t5_d_ff: int = MISSING
    t5_d_kv: int = MISSING
    mask_ratio: float = MISSING
    dropout: float = MISSING
    grad_clip_norm: float = MISSING


@dataclass
class TimesfmSchema:
    config_size: str = MISSING
    grad_clip_norm: float = MISSING
    normalization_mode: str = MISSING
    objective: str = MISSING


@dataclass
class TsfmConfig:
    model: str = MISSING
    device: str = MISSING
    condition: str = MISSING
    experiment_kind: str = MISSING
    scale_assignment: str | None = None
    scale_b_low: float = MISSING
    scale_b_high: float = MISSING
    seed: int = MISSING
    output_dir: str = MISSING
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    window_index: WindowIndexConfig = field(default_factory=WindowIndexConfig)
    dataset_weights: dict[str, float] | None = None
    train: TsfmTrainConfig = field(default_factory=TsfmTrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    chronos2: Chronos2Schema = field(default_factory=Chronos2Schema)
    moirai2: Moirai2Schema = field(default_factory=Moirai2Schema)
    moment: MomentSchema = field(default_factory=MomentSchema)
    timesfm: TimesfmSchema = field(default_factory=TimesfmSchema)


def validate_config(cfg: DictConfig, schema: type) -> DictConfig:
    """Validate a composed Hydra config against a structured schema."""
    validated = OmegaConf.merge(OmegaConf.structured(schema), cfg)
    if schema is ToyConfig:
        OmegaConf.to_container(validated, resolve=False, throw_on_missing=True)
    elif schema is TsfmConfig:
        common_fields = (
            "model",
            "device",
            "condition",
            "experiment_kind",
            "scale_assignment",
            "scale_b_low",
            "scale_b_high",
            "seed",
            "output_dir",
            "corpus",
            "window_index",
            "dataset_weights",
            "train",
            "wandb",
        )
        scalar_fields = {
            "model",
            "device",
            "condition",
            "experiment_kind",
            "scale_assignment",
            "scale_b_low",
            "scale_b_high",
            "seed",
            "output_dir",
            "dataset_weights",
        }
        for field_name in common_fields:
            if field_name in scalar_fields:
                if OmegaConf.is_missing(validated, field_name):
                    validated[field_name]
            else:
                OmegaConf.to_container(
                    validated[field_name], resolve=False, throw_on_missing=True
                )
        OmegaConf.to_container(
            validated[cfg.model], resolve=False, throw_on_missing=True
        )
    else:
        raise TypeError(f"unsupported config schema {schema!r}")
    return cfg

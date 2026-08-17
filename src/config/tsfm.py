"""Structured schema for the TSFM GiftEvalPretrain loss-space runs."""

from dataclasses import dataclass, field

from omegaconf import MISSING

from src.config.base import WandbConfig


@dataclass
class CorpusConfig:
    root: str = MISSING
    datasets: list[str] | None = None
    exclude: list[str] = field(default_factory=list)


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
    schedule_steps: int | None = None
    batch_size: int = MISSING
    lr: float = MISSING
    optimizer: str = MISSING
    deterministic: bool = MISSING
    schedule_seed: int = MISSING
    eval_every: int = MISSING
    eval_batches: int = MISSING
    eval_windows_per_dataset: int = MISSING
    checkpoint_every: int = MISSING
    resume_from: str | None = MISSING


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

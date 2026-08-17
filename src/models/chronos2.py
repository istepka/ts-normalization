"""Chronos-2 forecasting adapter for the GiftEvalPretrain loss study.

The model is the official Chronos-2 Small architecture, initialized from
scratch. Its native objective computes quantile loss after context-based
normalization. The counterfactual condition computes the same quantile loss
after reversing that normalization. Both conditions use the same forecasts,
windows, initialization, and optimization schedule.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from chronos.chronos2 import Chronos2CoreConfig, Chronos2Model

from src.data.gifteval import window_index as wi
from src.models import normalization

CHRONOS2_REVISION = "chronos-forecasting==2.3.1"


class IdentityInstanceNorm(torch.nn.Module):
    """Stands in for Chronos-2's `InstanceNorm`, including its `inverse`."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.use_arcsinh = False

    def forward(self, x: torch.Tensor, loc_scale=None):
        if loc_scale is None:
            loc = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
            scale = torch.ones(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
            loc_scale = (loc, scale)
        return x, loc_scale

    def inverse(self, x: torch.Tensor, loc_scale) -> torch.Tensor:
        return x


class Chronos2Normalization(normalization.BackboneNormalization):
    """Chronos-2's `InstanceNorm`, reproduced exactly.

    The arcsinh makes this the one backbone whose normalized and original
    spaces are related nonlinearly rather than by a per-window constant, so
    `ArcsinhStdScheme` carries the nonlinearity rather than the adapter.
    """

    normalized_condition = "chronos2_normalized"
    original_condition = "chronos2_original"

    def __init__(self, use_arcsinh: bool, eps: float = 1e-5):
        super().__init__(
            normalization.ArcsinhStdScheme(eps=eps, use_arcsinh=use_arcsinh)
        )

    @classmethod
    def from_model(cls, model: Chronos2Model) -> "Chronos2Normalization":
        """Rebuilds the scheme from a model that may already be disabled.

        `forward` receives a model but not the config. Both parameters survive
        `disable`, since `IdentityInstanceNorm` keeps `eps` and `use_arcsinh`
        is read off `chronos_config` rather than off the replaced module.
        """
        return cls(
            use_arcsinh=model.chronos_config.use_arcsinh,
            eps=model.instance_norm.eps,
        )

    def disable(self, model: Chronos2Model) -> None:
        model.instance_norm = IdentityInstanceNorm(eps=model.instance_norm.eps)


CONDITIONS = Chronos2Normalization.conditions()


@dataclass(frozen=True)
class Chronos2Config:
    context_length: int
    prediction_length: int
    patch_size: int
    d_model: int
    d_kv: int
    d_ff: int
    num_layers: int
    num_heads: int
    dropout_rate: float
    initializer_factor: float
    quantiles: tuple[float, ...]
    use_arcsinh: bool
    grad_clip_norm: float


def build_chronos2_model(config: Chronos2Config, seed: int) -> Chronos2Model:
    if config.context_length % config.patch_size != 0:
        raise ValueError("context_length must be divisible by patch_size")
    if config.prediction_length % config.patch_size != 0:
        raise ValueError("prediction_length must be divisible by patch_size")

    generator_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        core_config = Chronos2CoreConfig(
            d_model=config.d_model,
            d_kv=config.d_kv,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            initializer_factor=config.initializer_factor,
            attn_implementation="eager",
            chronos_config={
                "context_length": config.context_length,
                "input_patch_size": config.patch_size,
                "input_patch_stride": config.patch_size,
                "output_patch_size": config.patch_size,
                "quantiles": list(config.quantiles),
                "use_reg_token": True,
                "use_arcsinh": config.use_arcsinh,
                "max_output_patches": (config.prediction_length // config.patch_size),
                "time_encoding_scale": config.context_length,
            },
        )
        model = Chronos2Model(core_config)
        # `src/models/normalization.py` owns the transform now, so the
        # backbone must not normalize a second time.
        Chronos2Normalization(use_arcsinh=config.use_arcsinh).disable(model)
        return model
    finally:
        torch.random.set_rng_state(generator_state)


@dataclass
class Chronos2Batch:
    context: torch.Tensor
    context_valid: torch.Tensor
    target: torch.Tensor
    target_valid: torch.Tensor
    dataset: np.ndarray
    domain: np.ndarray
    frequency: np.ndarray
    scale: torch.Tensor

    def to(self, device: str) -> "Chronos2Batch":
        return Chronos2Batch(
            context=self.context.to(device),
            context_valid=self.context_valid.to(device),
            target=self.target.to(device),
            target_valid=self.target_valid.to(device),
            dataset=self.dataset,
            domain=self.domain,
            frequency=self.frequency,
            scale=self.scale.to(device),
        )


def make_batch(
    window_index: wi.WindowIndex,
    rows: pd.DataFrame,
    cache: wi.SeriesCache,
    *,
    scale_assignment: str | None = None,
    b_low: float = 1.0,
    b_high: float = 10.0,
) -> Chronos2Batch:
    context_length = window_index.config.context_length
    prediction_length = window_index.config.prediction_length
    contexts = np.full((len(rows), context_length), np.nan, dtype=np.float32)
    context_valid = np.zeros((len(rows), context_length), dtype=np.float32)
    targets = np.zeros((len(rows), prediction_length), dtype=np.float32)
    target_valid = np.zeros((len(rows), prediction_length), dtype=np.float32)
    scales = np.ones(len(rows), dtype=np.float32)

    for i, (_, row) in enumerate(rows.iterrows()):
        window = window_index.window_values(row, cache)
        context = window[:context_length].copy()
        target = window[context_length : context_length + prediction_length].copy()
        valid_context = ~np.isnan(context)
        valid_target = ~np.isnan(target)

        if scale_assignment is not None:
            b = window_index.scale_for(row, scale_assignment, b_low, b_high)
            mean = row["context_mean"]
            std = row["context_std"]
            context[valid_context] = b * (context[valid_context] - mean) / std
            target[valid_target] = b * (target[valid_target] - mean) / std
            scales[i] = b

        contexts[i] = context
        context_valid[i] = valid_context
        targets[i] = np.nan_to_num(target, nan=0.0)
        target_valid[i] = valid_target

    return Chronos2Batch(
        context=torch.from_numpy(contexts),
        context_valid=torch.from_numpy(context_valid),
        target=torch.from_numpy(targets),
        target_valid=torch.from_numpy(target_valid),
        dataset=rows["dataset"].to_numpy(),
        domain=rows["domain"].to_numpy(),
        frequency=rows["frequency"].to_numpy(),
        scale=torch.from_numpy(scales),
    )


def run_model(
    model: Chronos2Model, batch: Chronos2Batch
) -> tuple[torch.Tensor, torch.Tensor, normalization.TransformStats]:
    """Forward pass exposing the output in both spaces.

    The context is normalized here rather than inside `model.encode`, whose
    own instance norm `build_chronos2_model` has disabled. The returned
    statistics are what the caller reads the loss through.
    """
    backbone = Chronos2Normalization.from_model(model)
    z_context, stats = backbone.transform_input(batch.context, batch.context_valid)

    num_output_patches = batch.target.shape[1] // model.chronos_config.output_patch_size
    encoder_output, _, _, _ = model.encode(
        context=z_context,
        context_mask=batch.context_valid,
        num_output_patches=num_output_patches,
    )
    hidden = encoder_output.last_hidden_state[:, -num_output_patches:]
    projected = model.output_patch_embedding(hidden)
    batch_size = projected.shape[0]
    normalized = projected.view(
        batch_size,
        num_output_patches,
        model.num_quantiles,
        model.chronos_config.output_patch_size,
    )
    normalized = normalized.permute(0, 2, 1, 3).reshape(
        batch_size, model.num_quantiles, -1
    )
    return normalized, stats.inverse(normalized), stats


@dataclass
class Chronos2ForwardResult:
    loss_per_example: torch.Tensor
    normalized_mse: torch.Tensor
    mase: torch.Tensor
    degenerate: torch.Tensor
    original_point_forecast: torch.Tensor


def forward(
    model: Chronos2Model,
    batch: Chronos2Batch,
    condition: str,
) -> Chronos2ForwardResult:
    module = Chronos2Normalization.from_model(model).module(condition)

    normalized, original, stats = run_model(model, batch)
    # The condition is the whole difference: SIT scores in normalized space,
    # RevIN in original space, off the same forward pass.
    predictions, target = module.transform_target_and_output(
        normalized, batch.target, stats
    )
    loss = quantile.crps_quantile_loss(
        predictions,
        target,
        batch.target_valid,
        model.quantiles,
    )

    median_index = model.chronos_config.quantiles.index(0.5)
    original_point = original[:, median_index]
    standardized_error = (original_point - batch.target) / stats.scale.unsqueeze(-1)
    normalized_mse = pointwise.masked_mse(
        standardized_error,
        torch.zeros_like(standardized_error),
        batch.target_valid,
        reduction="none",
    )
    original_mae = pointwise.masked_mae(
        original_point,
        batch.target,
        batch.target_valid,
        reduction="none",
    )
    periods = torch.as_tensor(
        [seasonality.seasonal_period(frequency) for frequency in batch.frequency],
        device=batch.context.device,
    )
    naive_mae = forecast.seasonal_naive_mae(
        torch.nan_to_num(batch.context), batch.context_valid, periods
    )

    # Eligibility must not depend on the imposed scale, so the known b is
    # divided out to recover the pre-intervention scale before the check.
    base_scale = stats.scale / batch.scale
    degenerate = base_scale <= stats.scheme.scale_floor * (1.0 + 1e-6)
    nan = torch.tensor(float("nan"), device=batch.context.device)
    return Chronos2ForwardResult(
        loss_per_example=loss,
        normalized_mse=torch.where(degenerate, nan, normalized_mse),
        mase=torch.where(degenerate, nan, original_mae / naive_mae),
        degenerate=degenerate,
        original_point_forecast=original_point,
    )


def training_step_metrics(
    model: Chronos2Model,
    batch: Chronos2Batch,
    condition: str,
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float,
) -> dict:
    optimizer.zero_grad(set_to_none=True)
    result = forward(model, batch, condition)
    keep = ~result.degenerate
    if not bool(keep.any()):
        raise RuntimeError("every window in the batch has a degenerate scale")
    loss = result.loss_per_example[keep].mean()
    gradient_metrics = gradients.backward_with_safe_gradient_clipping(
        loss,
        model.parameters(),
        grad_clip_norm,
        tracked_parameters=model.output_patch_embedding.parameters(),
    )
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "loss_per_example": result.loss_per_example.detach().cpu().numpy(),
        "output_head_grad_norm": gradient_metrics["tracked_norm_before_clip"],
        "total_grad_norm_before_clip": gradient_metrics["total_norm_before_clip"],
        "total_grad_norm_after_clip": gradient_metrics["total_norm_after_clip"],
        "clipped": gradient_metrics["clipped"],
        "step_skipped": False,
        "degenerate_frac": float(result.degenerate.float().mean()),
        "dataset": batch.dataset,
        "domain": batch.domain,
        "frequency": batch.frequency,
    }


@dataclass
class Chronos2Forecaster:
    """`src.eval.protocol.Forecaster` over a trained Chronos-2 checkpoint."""

    model: Chronos2Model
    device: str
    context_length: int
    horizon: int
    quantiles: list[float]

    def predict(
        self, context: np.ndarray, valid: np.ndarray, freqs: list[str]
    ) -> np.ndarray:
        if context.shape[1] != self.context_length:
            raise ValueError(
                f"context is {context.shape[1]} wide, expected {self.context_length}"
            )
        # make_batch leaves missing context as NaN alongside the mask, so the
        # padded positions are restored to NaN rather than passed as zeros.
        masked = np.where(valid > 0, context, np.nan).astype(np.float32)
        batch_size = context.shape[0]
        batch = Chronos2Batch(
            context=torch.from_numpy(masked),
            context_valid=torch.from_numpy(valid.astype(np.float32)),
            target=torch.zeros(batch_size, self.horizon),
            target_valid=torch.zeros(batch_size, self.horizon),
            dataset=np.empty(batch_size, dtype=object),
            domain=np.empty(batch_size, dtype=object),
            frequency=np.asarray(freqs, dtype=object),
            scale=torch.ones(batch_size),
        ).to(self.device)

        with torch.no_grad():
            _, original, _ = run_model(self.model, batch)
        # run_model puts the quantile axis in the middle, [B, Q, H].
        return original.permute(0, 2, 1).float().cpu().numpy()


def build_forecaster(cfg, checkpoint_path, device: str) -> Chronos2Forecaster:
    model_config = Chronos2Config(**OmegaConf.to_container(cfg.chronos2, resolve=True))
    model = build_chronos2_model(model_config, seed=cfg.seed)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return Chronos2Forecaster(
        model=model,
        device=device,
        context_length=model_config.context_length,
        horizon=model_config.prediction_length,
        quantiles=list(model_config.quantiles),
    )


from omegaconf import OmegaConf

from src.data import seasonality
from src.losses import pointwise, quantile
from src.metrics import forecast
from src.training import gradients

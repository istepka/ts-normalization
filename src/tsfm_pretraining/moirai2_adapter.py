"""Moirai 2.0 forecasting adapter for the GiftEvalPretrain loss study.

The model is the official Moirai-2.0-R-small architecture (d_model=384,
d_ff=1024, 6 layers, patch_size=16, 9 quantile levels), initialized from
scratch. It natively predicts `num_predict_token` patches (4 patches = 64
steps for the official small config) from the last context token in a single
forward pass; this project's canonical GiftEvalPretrain window (512 context,
128 prediction) is reused unchanged for dataset/window comparability with the
other three TSFM arms, but Moirai 2.0 trains and evaluates on only the first
64 steps of each window's target -- see
notes/agentic_logs/2026-08-10-moirai2-loss-space-integration.md for why.

The native objective computes quantile (pinball) loss in the model's own
instance-normalized space (`moirai2_normalized`). The counterfactual
condition (`moirai2_original`) inverts that normalization with the same
loc/scale statistics and computes the same quantile loss in original units.
Both conditions share the same forward pass, windows, initialization, and
optimization schedule.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import losses as L
from . import window_index as wi
from .vendor.moirai2 import Moirai2Module

MOIRAI2_REVISION = (
    "uni2ts==2.0.0 (vendored, see src/tsfm_pretraining/vendor/moirai2/REVISION)"
)
CONDITIONS = ("moirai2_normalized", "moirai2_original")


@dataclass(frozen=True)
class Moirai2Config:
    context_length: int
    predict_horizon: int
    patch_size: int
    d_model: int
    d_ff: int
    num_layers: int
    max_seq_len: int
    attn_dropout_p: float
    dropout_p: float
    scaling: bool
    quantile_levels: tuple[float, ...]
    grad_clip_norm: float

    @property
    def num_predict_token(self) -> int:
        return self.predict_horizon // self.patch_size

    @property
    def context_token_length(self) -> int:
        return self.context_length // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.context_token_length + self.num_predict_token


def build_moirai2_model(config: Moirai2Config, seed: int) -> Moirai2Module:
    if config.context_length % config.patch_size != 0:
        raise ValueError("context_length must be divisible by patch_size")
    if config.predict_horizon % config.patch_size != 0:
        raise ValueError("predict_horizon must be divisible by patch_size")
    if 0.5 not in config.quantile_levels:
        raise ValueError("quantile_levels must include the median (0.5)")
    if config.num_patches > config.max_seq_len:
        raise ValueError(
            f"num_patches ({config.num_patches}) exceeds max_seq_len "
            f"({config.max_seq_len})"
        )

    generator_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        return Moirai2Module(
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            patch_size=config.patch_size,
            max_seq_len=config.max_seq_len,
            attn_dropout_p=config.attn_dropout_p,
            dropout_p=config.dropout_p,
            scaling=config.scaling,
            num_predict_token=config.num_predict_token,
            quantile_levels=config.quantile_levels,
        )
    finally:
        torch.random.set_rng_state(generator_state)


@dataclass
class Moirai2Batch:
    target: torch.Tensor  # [B, num_patches, patch_size], original units
    observed_mask: torch.Tensor  # [B, num_patches, patch_size], bool
    sample_id: torch.Tensor  # [B, num_patches], long
    time_id: torch.Tensor  # [B, num_patches], long
    variate_id: torch.Tensor  # [B, num_patches], long
    prediction_mask: torch.Tensor  # [B, num_patches], bool
    dataset: np.ndarray
    domain: np.ndarray
    frequency: np.ndarray
    scale: torch.Tensor  # controlled-scale factor b applied in make_batch

    def to(self, device: str) -> "Moirai2Batch":
        return Moirai2Batch(
            target=self.target.to(device),
            observed_mask=self.observed_mask.to(device),
            sample_id=self.sample_id.to(device),
            time_id=self.time_id.to(device),
            variate_id=self.variate_id.to(device),
            prediction_mask=self.prediction_mask.to(device),
            dataset=self.dataset,
            domain=self.domain,
            frequency=self.frequency,
            scale=self.scale.to(device),
        )


def make_batch(
    window_index: wi.WindowIndex,
    rows: pd.DataFrame,
    cache: wi.SeriesCache,
    config: Moirai2Config,
    *,
    scale_assignment: str | None = None,
    b_low: float = 1.0,
    b_high: float = 10.0,
) -> Moirai2Batch:
    seq_len = config.context_length + config.predict_horizon
    context_patches = config.context_token_length
    num_patches = config.num_patches
    n = len(rows)

    sequences = np.full((n, seq_len), np.nan, dtype=np.float32)
    scales = np.ones(n, dtype=np.float32)

    for i, (_, row) in enumerate(rows.iterrows()):
        window = window_index.window_values(row, cache)
        sequence = window[:seq_len].copy()
        valid = ~np.isnan(sequence)

        if scale_assignment is not None:
            b = window_index.scale_for(row, scale_assignment, b_low, b_high)
            mean = row["context_mean"]
            std = row["context_std"]
            sequence[valid] = b * (sequence[valid] - mean) / std
            scales[i] = b

        sequences[i] = sequence

    observed = ~np.isnan(sequences)
    sequences = np.nan_to_num(sequences, nan=0.0)

    target = torch.from_numpy(sequences).reshape(n, num_patches, config.patch_size)
    observed_mask = torch.from_numpy(observed).reshape(
        n, num_patches, config.patch_size
    )
    sample_id = torch.ones(n, num_patches, dtype=torch.long)
    time_id = torch.arange(num_patches, dtype=torch.long).unsqueeze(0).expand(n, -1)
    variate_id = torch.zeros(n, num_patches, dtype=torch.long)
    prediction_mask = torch.zeros(n, num_patches, dtype=torch.bool)
    prediction_mask[:, context_patches:] = True

    return Moirai2Batch(
        target=target,
        observed_mask=observed_mask,
        sample_id=sample_id,
        time_id=time_id.clone(),
        variate_id=variate_id,
        prediction_mask=prediction_mask,
        dataset=rows["dataset"].to_numpy(),
        domain=rows["domain"].to_numpy(),
        frequency=rows["frequency"].to_numpy(),
        scale=torch.from_numpy(scales),
    )


def run_model(
    model: Moirai2Module, batch: Moirai2Batch, config: Moirai2Config
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs the model and returns per-example quantities in patch order.

    normalized_preds: [B, num_quantiles, predict_horizon]
    normalized_target: [B, predict_horizon]
    loc, scale: [B], the context statistics used to normalize this window
    """
    context_mask = batch.observed_mask * ~batch.prediction_mask.unsqueeze(-1)
    loc, scale = model.scaler(
        batch.target, context_mask, batch.sample_id, batch.variate_id
    )
    preds, scaled_target = model(
        batch.target,
        batch.observed_mask,
        batch.sample_id,
        batch.time_id,
        batch.variate_id,
        batch.prediction_mask,
        training_mode=True,
    )

    batch_size = preds.shape[0]
    pred_position = config.context_token_length - 1
    raw_preds = preds[:, pred_position].view(
        batch_size, model.num_predict_token, model.num_quantiles, model.patch_size
    )
    normalized_preds = raw_preds.permute(0, 2, 1, 3).reshape(
        batch_size, model.num_quantiles, config.predict_horizon
    )
    normalized_target = scaled_target[:, config.context_token_length :].reshape(
        batch_size, config.predict_horizon
    )
    return normalized_preds, normalized_target, loc[:, 0, 0], scale[:, 0, 0]


def _quantile_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    target = target.unsqueeze(1)
    valid = valid.unsqueeze(1)
    quantiles = quantiles.view(1, -1, 1)
    loss = 2.0 * torch.abs(
        (target - predictions)
        * ((target <= predictions).to(predictions.dtype) - quantiles)
    )
    return (loss * valid).mean(dim=-1).sum(dim=-1)


@dataclass
class Moirai2ForwardResult:
    loss_per_example: torch.Tensor
    normalized_mse: torch.Tensor
    mase: torch.Tensor
    degenerate: torch.Tensor
    original_point_forecast: torch.Tensor


def forward(
    model: Moirai2Module,
    batch: Moirai2Batch,
    condition: str,
    config: Moirai2Config,
) -> Moirai2ForwardResult:
    if condition not in CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r}, must be one of {CONDITIONS}"
        )

    normalized_preds, normalized_target, loc, scale = run_model(model, batch, config)
    batch_size = normalized_preds.shape[0]
    context_patches = config.context_token_length

    future_target = batch.target[:, context_patches:].reshape(
        batch_size, config.predict_horizon
    )
    future_valid = batch.observed_mask[:, context_patches:].reshape(
        batch_size, config.predict_horizon
    )

    if condition == "moirai2_normalized":
        predictions = normalized_preds
        target = normalized_target
    else:
        predictions = normalized_preds * scale.view(-1, 1, 1) + loc.view(-1, 1, 1)
        target = future_target
    quantiles = torch.as_tensor(
        model.quantile_levels, device=predictions.device, dtype=predictions.dtype
    )
    loss = _quantile_loss(predictions, target, future_valid, quantiles)

    median_index = model.quantile_levels.index(0.5)
    original_point = normalized_preds[:, median_index] * scale.unsqueeze(-1) + (
        loc.unsqueeze(-1)
    )
    standardized_error = (original_point - future_target) / scale.unsqueeze(-1)
    normalized_mse = L.masked_mse(
        standardized_error,
        torch.zeros_like(standardized_error),
        future_valid,
        reduction="none",
    )
    original_mae = L.masked_mae(
        original_point, future_target, future_valid, reduction="none"
    )
    periods = torch.as_tensor(
        [L.seasonal_period(frequency) for frequency in batch.frequency],
        device=batch.target.device,
    )
    context_raw = batch.target[:, :context_patches].reshape(
        batch_size, config.context_length
    )
    context_valid = batch.observed_mask[:, :context_patches].reshape(
        batch_size, config.context_length
    )
    naive_mae = L.seasonal_naive_mae(context_raw, context_valid, periods)

    base_scale = scale / batch.scale
    degenerate_floor = model.scaler.minimum_scale**0.5
    degenerate = base_scale <= degenerate_floor * (1.0 + 1e-6)
    nan = torch.tensor(float("nan"), device=batch.target.device)
    return Moirai2ForwardResult(
        loss_per_example=loss,
        normalized_mse=torch.where(degenerate, nan, normalized_mse),
        mase=torch.where(degenerate, nan, original_mae / naive_mae),
        degenerate=degenerate,
        original_point_forecast=original_point,
    )


def training_step_metrics(
    model: Moirai2Module,
    batch: Moirai2Batch,
    condition: str,
    config: Moirai2Config,
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float,
) -> dict:
    optimizer.zero_grad(set_to_none=True)
    result = forward(model, batch, condition, config)
    keep = ~result.degenerate
    if not bool(keep.any()):
        raise RuntimeError("every window in the batch has a degenerate scale")
    loss = result.loss_per_example[keep].mean()
    gradient_metrics = L.backward_with_safe_gradient_clipping(
        loss,
        model.parameters(),
        grad_clip_norm,
        tracked_parameters=model.out_proj.parameters(),
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

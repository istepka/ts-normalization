"""Project adapter around the vendored MOMENT model for the loss-space study.

Wraps `vendor.moment.MOMENT` (pinned revision recorded in
vendor/moment/REVISION) with:

- deterministic masked-window batches, built from the shared WindowIndex's
  per-window `mask_seed` via `torch.random.fork_rng` so the global RNG state
  used for model init / data order elsewhere is never disturbed
- both loss-space conditions from the plan:
  - `moment_normalized`: MOMENT's native objective, MSE between the
    normalized decoder output and the normalized target (RevIN-normalized
    with the same statistics used internally by the forward pass).
  - `moment_original`: identical forward pass and mask, but MSE is computed
    between the denormalized (original-space) reconstruction and the
    original-space target.
- masked vs unmasked error, gradient norms before/after clipping, and the
  per-source dispersion metrics from losses.py

The expected controlled-scale gradient ratio between two scale assignments
b_low/b_high is approximately (b_high / b_low) ** 2 for MSE (see the plan's
"MOMENT stage" section).
"""

from argparse import Namespace
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.data.gifteval import window_index as wi
from src.models.vendor.moment import MOMENT

MOMENT_REVISION = "38f7310ad594100747ca2a8357e9c7ca7d323e0e"
CONDITIONS = ("moment_normalized", "moment_original")


@dataclass(frozen=True)
class MomentConfig:
    context_length: int = 512
    patch_len: int = 8
    d_model: int = 256
    t5_num_layers: int = 4
    t5_num_heads: int = 8
    t5_d_ff: int = 512
    t5_d_kv: int = 32
    mask_ratio: float = 0.3
    dropout: float = 0.1
    grad_clip_norm: float = 1.0

    def __post_init__(self):
        if self.context_length % self.patch_len != 0:
            raise ValueError("context_length must be a multiple of patch_len")
        # No d_model/t5_num_heads divisibility constraint: T5 attention
        # projects d_model into num_heads*d_kv independently (head dim is
        # d_kv, not d_model/num_heads), so e.g. flan-t5-small's real
        # d_model=512, num_heads=6 (512/6 is not integral) is a valid config.


def build_moment_model(config: MomentConfig, seed: int) -> MOMENT:
    generator_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        t5_config = dict(
            d_model=config.d_model,
            d_ff=config.t5_d_ff,
            num_layers=config.t5_num_layers,
            num_heads=config.t5_num_heads,
            d_kv=config.t5_d_kv,
            vocab_size=32,
            dropout_rate=config.dropout,
            feed_forward_proj="relu",
        )
        model_config = Namespace(
            task_name="reconstruction",
            seq_len=config.context_length,
            patch_len=config.patch_len,
            patch_stride_len=config.patch_len,
            d_model=config.d_model,
            transformer_backbone="google/flan-t5-small",
            transformer_type="encoder_only",
            t5_config=t5_config,
            randomly_initialize_backbone=True,
            mask_ratio=config.mask_ratio,
            dropout=config.dropout,
            enable_gradient_checkpointing=False,
            freeze_embedder=False,
            freeze_encoder=False,
            freeze_head=False,
            revin_affine=False,
        )
        return MOMENT(model_config)
    finally:
        torch.random.set_rng_state(generator_state)


@dataclass
class MomentBatch:
    x_enc: torch.Tensor  # [B, 1, context_length], original space, NaNs zero-filled
    input_mask: torch.Tensor  # [B, context_length], 1 = observed (not missing/padding)
    dataset: np.ndarray  # [B] str
    domain: np.ndarray  # [B] str
    frequency: np.ndarray  # [B] str
    scale: (
        torch.Tensor
    )  # [B], scale applied to this window (1.0 for natural-mixture runs)
    batch_seed: int

    def to(self, device: str) -> "MomentBatch":
        return MomentBatch(
            x_enc=self.x_enc.to(device),
            input_mask=self.input_mask.to(device),
            dataset=self.dataset,
            domain=self.domain,
            frequency=self.frequency,
            scale=self.scale.to(device),
            batch_seed=self.batch_seed,
        )


def make_batch(
    window_index: wi.WindowIndex,
    rows: pd.DataFrame,
    cache: wi.SeriesCache,
    *,
    scale_assignment: str | None = None,
    b_low: float = 1.0,
    b_high: float = 10.0,
) -> MomentBatch:
    """Builds one MomentBatch from a slice of the window index table.

    If `scale_assignment` is given ('A' or 'B'), each window's context is
    rescaled by its assigned scale-swap factor (applied in original space,
    before any RevIN normalization inside the model. The raw context is first
    standardized with its stored causal context statistics and then multiplied
    by `b`. This isolates the controlled scale from raw levels and avoids
    float32 cancellation on large-valued series.
    """
    context_length = window_index.config.context_length
    contexts = np.zeros((len(rows), context_length), dtype=np.float32)
    masks = np.zeros((len(rows), context_length), dtype=np.float32)
    scales = np.ones(len(rows), dtype=np.float32)

    for i, (_, row) in enumerate(rows.iterrows()):
        window = window_index.window_values(row, cache)
        context = window[:context_length]
        valid = ~np.isnan(context)
        context = np.nan_to_num(context, nan=0.0)
        if scale_assignment is not None:
            b = window_index.scale_for(row, scale_assignment, b_low, b_high)
            mean = row["context_mean"]
            std = row["context_std"]
            context[valid] = b * (context[valid] - mean) / std
            context[~valid] = 0.0
            scales[i] = b
        contexts[i] = context
        masks[i] = valid

    batch_seed = wi.stable_seed(*sorted(int(s) for s in rows["mask_seed"]))
    return MomentBatch(
        x_enc=torch.from_numpy(contexts).unsqueeze(1),
        input_mask=torch.from_numpy(masks),
        dataset=rows["dataset"].to_numpy(),
        domain=rows["domain"].to_numpy(),
        frequency=rows["frequency"].to_numpy(),
        scale=torch.from_numpy(scales),
        batch_seed=batch_seed,
    )


@dataclass
class MomentForwardResult:
    per_example_loss_masked: (
        torch.Tensor
    )  # [B], the training objective's per-example loss
    per_example_loss_unmasked: (
        torch.Tensor
    )  # [B], same space, evaluated over ALL positions
    normalized_mse: torch.Tensor  # [B], masked, always in normalized space
    original_mse: torch.Tensor  # [B], masked, always in original space
    mase: torch.Tensor  # [B], masked original-space MAE / seasonal-naive MAE
    reconstruction: torch.Tensor  # [B, 1, context_length], original space
    pretrain_mask: torch.Tensor  # [B, context_length], 0 = reconstructed position


def forward(model: MOMENT, batch: MomentBatch, condition: str) -> MomentForwardResult:
    if condition not in CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r}, must be one of {CONDITIONS}"
        )

    # Masking.generate_mask draws on batch.x_enc.device's RNG stream (see
    # vendor/moment/utils/masking.py); fork_rng must cover that device too or
    # a GPU run silently uses (and permanently advances) the unseeded global
    # CUDA RNG instead of the deterministic per-window batch_seed.
    fork_devices = [batch.x_enc.device] if batch.x_enc.device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(batch.batch_seed)
        out = model(x_enc=batch.x_enc, input_mask=batch.input_mask)

    reconstructed_mask = (
        1.0 - out.pretrain_mask
    )  # 0 in pretrain_mask means "reconstruct this"
    # [B, context_length] -> [B, 1, context_length] to match the channel dim
    # on reconstruction/x_enc; masked_mse's mask must already be broadcastable
    # to pred/target's shape, so an un-unsqueezed [B, L] mask against a
    # [B, 1, L] pred would silently broadcast across a spurious batch x batch
    # cross product instead of a per-example mask.
    train_mask = (reconstructed_mask * batch.input_mask).unsqueeze(1)
    input_mask_ch = batch.input_mask.unsqueeze(1)

    normalized_recon = out.metadata["normalized_reconstruction"]
    revin_mean = out.metadata["revin_mean"]
    revin_stdev = out.metadata["revin_stdev"]
    normalized_target = (batch.x_enc - revin_mean) / revin_stdev

    normalized_mse = pointwise.masked_mse(
        normalized_recon, normalized_target, train_mask, reduction="none"
    )
    original_mse = pointwise.masked_mse(
        out.reconstruction, batch.x_enc, train_mask, reduction="none"
    )
    unmasked_mse = pointwise.masked_mse(
        out.reconstruction, batch.x_enc, input_mask_ch, reduction="none"
    )

    per_example = normalized_mse if condition == "moment_normalized" else original_mse

    # MASE numerator and denominator both live in the original space, so the
    # controlled scale multiplier cancels and the metric stays comparable
    # across scale assignments and across conditions.
    original_mae = pointwise.masked_mae(
        out.reconstruction, batch.x_enc, train_mask, reduction="none"
    )
    periods = torch.as_tensor(
        [seasonality.seasonal_period(f) for f in batch.frequency],
        device=batch.x_enc.device,
    )
    naive_mae = forecast.seasonal_naive_mae(
        batch.x_enc.squeeze(1), batch.input_mask, periods
    )
    mase = original_mae / naive_mae

    return MomentForwardResult(
        per_example_loss_masked=per_example,
        per_example_loss_unmasked=unmasked_mse,
        normalized_mse=normalized_mse,
        original_mse=original_mse,
        mase=mase,
        reconstruction=out.reconstruction,
        pretrain_mask=out.pretrain_mask,
    )


def training_step_metrics(
    model: MOMENT,
    batch: MomentBatch,
    condition: str,
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float,
) -> dict:
    """Runs one forward/backward/optimizer step and returns the metrics required
    by the plan's "MOMENT metrics" section for this batch (dataset/domain/
    frequency breakdowns are computed by the caller from the returned arrays,
    since a single training step spans many sources)."""
    optimizer.zero_grad(set_to_none=True)
    result = forward(model, batch, condition)
    loss = result.per_example_loss_masked.mean()
    gradient_metrics = gradients.backward_with_safe_gradient_clipping(
        loss,
        model.parameters(),
        grad_clip_norm,
    )
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "per_example_loss_masked": result.per_example_loss_masked.detach()
        .cpu()
        .numpy(),
        "per_example_loss_unmasked": result.per_example_loss_unmasked.detach()
        .cpu()
        .numpy(),
        "normalized_mse": result.normalized_mse.detach().cpu().numpy(),
        "original_mse": result.original_mse.detach().cpu().numpy(),
        "grad_norm_before_clip": gradient_metrics["total_norm_before_clip"],
        "grad_norm_after_clip": gradient_metrics["total_norm_after_clip"],
        "clipped": gradient_metrics["clipped"],
        "step_skipped": False,
        "dataset": batch.dataset,
        "domain": batch.domain,
        "frequency": batch.frequency,
    }


from src.data import seasonality
from src.losses import pointwise
from src.metrics import forecast
from src.training import gradients

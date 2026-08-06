"""Reduced TimesFM-1.0 configs and loss-space adapter for the pretraining study.

Wraps `vendor.timesfm_v1.pytorch_patched_decoder.PatchedTimeSeriesDecoder`
(Google's own PyTorch port of the legacy TimesFM-1.0 architecture, pinned
revision recorded in vendor/timesfm_v1/REVISION, vendored byte-for-byte and
otherwise untouched) with reduced ~17M (smoke test) and ~70M (primary run)
parameter configs, per the plan's "Use a reduced configuration for the main
run" instruction.

A canonical window (context_length raw points followed by prediction_length
raw points, prediction_length == TimesFMConfig.horizon_len == 128) is fed to
the decoder as context; the model's last patch position forecasts exactly the
held-out prediction_length window, which is what training loss is computed
against (this mirrors how `PatchedTimeSeriesDecoder.decode` already reads the
last patch's output as the forecast, so training and the model's own
inference path agree).

The two loss-space conditions:
- `timesfm_native_original`: MSE (point head) and pinball (quantile heads)
  computed after the model's own instance-normalization inverse transform,
  i.e. in the corpus's original scale. This is the released MSE/pinball path.
- `timesfm_normalized`: the same terms computed before the inverse
  transform, directly on the normalized decoder output and a target
  normalized with the same per-window instance-norm statistics.

`PatchedTimeSeriesDecoder.forward` only returns the already-inverse-transformed
output, so `run_decoder` below re-derives the pre-inverse-transform output by
calling the same submodules `forward` calls internally
(`_preprocess_input`, `stacked_transformer`, `horizon_ff_layer`,
`_reverse_transform`) in the same order, rather than modifying the vendored
file. Expected controlled-scale scaling: MSE contribution ~ b**2, pinball
contribution ~ b (see the plan's "TimesFM stage" section).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from pandas.tseries.frequencies import to_offset

from . import losses as L
from . import window_index as wi
from .vendor.timesfm_v1.pytorch_patched_decoder import (
    PatchedTimeSeriesDecoder,
    TimesFMConfig,
)

TIMESFM_REVISION = "705685c9122eeecc53e57285e44598c3453acb60"
CONDITIONS = ("timesfm_native_original", "timesfm_normalized")

# ~17.7M params: smoke-test config (see notes/agentic_logs for the search that
# picked these dims against PatchedTimeSeriesDecoder's actual parameter count).
CONFIG_17M = TimesFMConfig(
    num_layers=10,
    num_heads=8,
    num_kv_heads=8,
    hidden_size=512,
    intermediate_size=512,
    head_dim=64,
    patch_len=32,
    horizon_len=128,
)
# ~70.9M params: primary-run config. head_dim=80 matches the official 200M
# checkpoint's head_dim, hidden_size/num_heads scaled down from 1280/16.
CONFIG_70M = TimesFMConfig(
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
    hidden_size=960,
    intermediate_size=960,
    head_dim=80,
    patch_len=32,
    horizon_len=128,
)


def build_timesfm_model(config: TimesFMConfig, seed: int) -> PatchedTimeSeriesDecoder:
    generator_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = PatchedTimeSeriesDecoder(config)
        _initialize_attention_scaling(model)
        return model
    finally:
        torch.random.set_rng_state(generator_state)


def _initialize_attention_scaling(model: PatchedTimeSeriesDecoder) -> None:
    """TimesFMAttention.scaling is allocated with `torch.empty` and never
    initialized in the vendored code (see vendor/timesfm_v1/REVISION) --
    harmless for Google's released checkpoint, which overwrites every
    parameter on load, but genuine uninitialized memory when training from a
    random init as this project does: values observed in practice range up
    to ~1e30, which explode through softplus into the attention scale and
    intermittently produce NaN logits. Zero is `_per_dim_scaling`'s implied
    neutral value (softplus(0) = ln(2), a bounded, reasonable per-head scale).
    """
    for module in model.modules():
        if hasattr(module, "scaling") and isinstance(
            module.scaling, torch.nn.Parameter
        ):
            torch.nn.init.zeros_(module.scaling)


# GiftEvalPretrain stores gluonts/old-pandas frequency aliases (e.g. "M",
# "5T", "A-DEC") that newer pandas rejects in favor of "ME"/"min"/"YE-DEC".
_OLD_TO_NEW_FREQ_BASE = {
    "Y": "YE",
    "A": "YE",
    "Q": "QE",
    "M": "ME",
    "H": "h",
    "T": "min",
    "S": "s",
    "U": "us",
}


def _parse_offset(freq: str):
    base, _, anchor = freq.partition("-")
    split = next((i for i, c in enumerate(base) if not c.isdigit()), len(base))
    mult, code = base[:split], base[split:]
    new_code = _OLD_TO_NEW_FREQ_BASE.get(code, code)
    anchor_suffix = f"-{anchor}" if anchor else ""
    return to_offset(f"{mult}{new_code}{anchor_suffix}")


def frequency_bucket(freq: str) -> int:
    """Maps a pandas-style frequency string to TimesFM's 3-way frequency
    embedding bucket (0 = sub-daily, 1 = daily/weekly, 2 = monthly or coarser).

    This is our own bucketing for training the freq_emb from scratch on
    GiftEvalPretrain; it is not claimed to reproduce Google's original
    checkpoint's bucket semantics, which don't matter here since freq_emb is
    randomly initialized and learned fresh.
    """
    offset = _parse_offset(freq)
    try:
        seconds = offset.nanos / 1e9
    except ValueError:
        return 1 if type(offset).__name__ == "Week" else 2
    if seconds <= 3600:
        return 0
    if seconds <= 7 * 86400:
        return 1
    return 2


@dataclass
class TimesFMBatch:
    context: torch.Tensor  # [B, context_length], original space, NaNs zero-filled
    context_padding: (
        torch.Tensor
    )  # [B, context_length], 1 = missing/padding (TimesFM convention)
    target: torch.Tensor  # [B, horizon_len], original space
    target_valid: torch.Tensor  # [B, horizon_len], 1 = valid target point
    freq: torch.Tensor  # [B, 1], long
    dataset: np.ndarray
    domain: np.ndarray
    frequency: np.ndarray
    scale: torch.Tensor

    def to(self, device: str) -> "TimesFMBatch":
        return TimesFMBatch(
            context=self.context.to(device),
            context_padding=self.context_padding.to(device),
            target=self.target.to(device),
            target_valid=self.target_valid.to(device),
            freq=self.freq.to(device),
            dataset=self.dataset,
            domain=self.domain,
            frequency=self.frequency,
            scale=self.scale.to(device),
        )


def make_batch(
    window_index: wi.WindowIndex,
    rows: pd.DataFrame,
    cache: wi.SeriesCache,
    horizon_len: int,
    *,
    scale_assignment: str | None = None,
    b_low: float = 1.0,
    b_high: float = 10.0,
) -> TimesFMBatch:
    if window_index.config.prediction_length != horizon_len:
        raise ValueError(
            "window index prediction_length must equal the model's horizon_len "
            f"({window_index.config.prediction_length} != {horizon_len})"
        )
    context_length = window_index.config.context_length
    contexts = np.zeros((len(rows), context_length), dtype=np.float32)
    context_pad = np.zeros((len(rows), context_length), dtype=np.float32)
    targets = np.zeros((len(rows), horizon_len), dtype=np.float32)
    target_valid = np.zeros((len(rows), horizon_len), dtype=np.float32)
    scales = np.ones(len(rows), dtype=np.float32)
    freqs = np.zeros(len(rows), dtype=np.int64)

    for i, (_, row) in enumerate(rows.iterrows()):
        window = window_index.window_values(row, cache)
        context = window[:context_length]
        target = window[context_length : context_length + horizon_len]
        context_valid = ~np.isnan(context)
        tgt_valid = ~np.isnan(target)
        context = np.nan_to_num(context, nan=0.0)
        target = np.nan_to_num(target, nan=0.0)

        if scale_assignment is not None:
            b = window_index.scale_for(row, scale_assignment, b_low, b_high)
            mean = row["context_mean"]
            context = mean + b * (context - mean)
            target = mean + b * (target - mean)
            scales[i] = b

        contexts[i] = context
        context_pad[i] = 1.0 - context_valid
        targets[i] = target
        target_valid[i] = tgt_valid
        freqs[i] = frequency_bucket(row["frequency"])

    return TimesFMBatch(
        context=torch.from_numpy(contexts),
        context_padding=torch.from_numpy(context_pad),
        target=torch.from_numpy(targets),
        target_valid=torch.from_numpy(target_valid),
        freq=torch.from_numpy(freqs).unsqueeze(1),
        dataset=rows["dataset"].to_numpy(),
        domain=rows["domain"].to_numpy(),
        frequency=rows["frequency"].to_numpy(),
        scale=torch.from_numpy(scales),
    )


def run_decoder(model: PatchedTimeSeriesDecoder, batch: TimesFMBatch):
    """Forward pass exposing both the normalized (pre-inverse-transform) and
    original-space (post-inverse-transform) last-patch output, following the
    exact same submodule sequence as `PatchedTimeSeriesDecoder.forward`."""
    num_outputs = len(model.config.quantiles) + 1
    model_input, patched_padding, stats, _ = model._preprocess_input(
        input_ts=batch.context, input_padding=batch.context_padding
    )
    f_emb = model.freq_emb(batch.freq)
    model_input = model_input + f_emb
    model_output = model.stacked_transformer(model_input, patched_padding)

    output_ts = model.horizon_ff_layer(model_output)
    b, n, _ = output_ts.shape
    normalized_out = output_ts.view(b, n, model.config.horizon_len, num_outputs)
    original_out = model._reverse_transform(normalized_out, stats)

    # Last patch position forecasts the held-out horizon.
    return normalized_out[:, -1], original_out[:, -1], stats


@dataclass
class TimesFMForwardResult:
    mse_per_example: torch.Tensor  # [B]
    pinball_per_example: torch.Tensor  # [B]
    loss_per_example: torch.Tensor  # [B], mse + mean(pinball), in `condition`'s space
    original_point_forecast: torch.Tensor  # [B, horizon_len]


def forward(
    model: PatchedTimeSeriesDecoder, batch: TimesFMBatch, condition: str
) -> TimesFMForwardResult:
    if condition not in CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r}, must be one of {CONDITIONS}"
        )

    normalized_out, original_out, (mu, sigma) = run_decoder(model, batch)
    quantiles = model.config.quantiles

    if condition == "timesfm_native_original":
        point = original_out[..., 0]
        quantile_pred = original_out[..., 1:]
        target = batch.target
    else:
        point = normalized_out[..., 0]
        quantile_pred = normalized_out[..., 1:]
        target = (batch.target - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)

    mse = L.masked_mse(point, target, batch.target_valid, reduction="none")
    pinball = _masked_pinball(quantile_pred, target, batch.target_valid, quantiles)

    return TimesFMForwardResult(
        mse_per_example=mse,
        pinball_per_example=pinball,
        loss_per_example=mse + pinball,
        original_point_forecast=original_out[..., 0],
    )


def _masked_pinball(
    pred_quantiles: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    quantiles: list[float],
) -> torch.Tensor:
    q = torch.as_tensor(
        quantiles, dtype=pred_quantiles.dtype, device=pred_quantiles.device
    )
    diff = target.unsqueeze(-1) - pred_quantiles
    loss = torch.maximum(q * diff, (q - 1.0) * diff)  # [B, horizon, Q]
    loss = loss * valid.unsqueeze(-1)
    denom = valid.sum(dim=1).clamp_min(1.0) * len(quantiles)
    return loss.sum(dim=(1, 2)) / denom


def training_step_metrics(
    model: PatchedTimeSeriesDecoder,
    batch: TimesFMBatch,
    condition: str,
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float,
) -> dict:
    optimizer.zero_grad(set_to_none=True)
    result = forward(model, batch, condition)
    loss = result.loss_per_example.mean()
    loss.backward()

    mse_params = list(model.horizon_ff_layer.parameters())
    mse_grad_norm = L.grad_norm(mse_params)
    total_grad_norm_before = L.grad_norm(model.parameters())
    clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    total_grad_norm_after = L.grad_norm(model.parameters())
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "mse_per_example": result.mse_per_example.detach().cpu().numpy(),
        "pinball_per_example": result.pinball_per_example.detach().cpu().numpy(),
        "mse_to_pinball_loss_ratio": float(
            result.mse_per_example.detach().mean()
            / result.pinball_per_example.detach().mean().clamp_min(1e-12)
        ),
        "mse_grad_norm": mse_grad_norm,
        "total_grad_norm_before_clip": total_grad_norm_before,
        "total_grad_norm_after_clip": total_grad_norm_after,
        "clipped": bool(float(clipped_norm) > grad_clip_norm),
        "dataset": batch.dataset,
        "domain": batch.domain,
        "frequency": batch.frequency,
    }

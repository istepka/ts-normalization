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

import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.data.gifteval import window_index as wi
from src.models import normalization
from src.models.vendor.timesfm_v1.pytorch_patched_decoder import (
    PatchedTimeSeriesDecoder,
    TimesFMConfig,
    _shift_padded_seq,
)

TIMESFM_REVISION = "705685c9122eeecc53e57285e44598c3453acb60"
NORMALIZATION_MODES = ("first_patch", "whole_context")
OBJECTIVES = ("mse", "combined")


class TimesFMScheme(normalization.NormalizationScheme):
    """TimesFM's instance norm, in either of the two modes this repo uses.

    This scheme stays in the adapter rather than in `normalization.py` because
    it is entangled with TimesFM's data encoding rather than being a general
    statistic. Padding is carried in-band as the sentinel value `pad_val`, and
    `forward` has to re-stamp it after scaling, which is upstream's behavior.

    `first_patch` reproduces vendored `_masked_mean_std`, which takes the
    statistics of the first patch holding more than three unpadded values.
    `whole_context` takes them over the entire valid context instead, which is
    this repo's causal variant (see `_preprocess_whole_context`).
    """

    def __init__(
        self,
        mode: str = "first_patch",
        patch_len: int = 32,
        tolerance: float = 1e-6,
        pad_val: float = 1123581321.0,
    ):
        if mode not in NORMALIZATION_MODES:
            raise ValueError(f"mode must be one of {NORMALIZATION_MODES}, got {mode!r}")
        self.mode = mode
        self.patch_len = patch_len
        self.tolerance = tolerance
        self.pad_val = pad_val
        self.scale_floor = tolerance

    def statistics(
        self, source: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = source.shape[0]
        patched = source.view(batch_size, -1, self.patch_len)
        pads = (1.0 - valid).view(batch_size, -1, self.patch_len)
        patched = torch.where(torch.abs(pads - 1.0) < self.tolerance, 0.0, patched)
        pads = torch.where(
            torch.abs(patched - self.pad_val) < self.tolerance,
            torch.ones_like(pads),
            pads,
        )

        if self.mode == "whole_context":
            unpadded = 1.0 - pads
            count = unpadded.sum(dim=(1, 2)).clamp_min(1.0)
            loc = (patched * unpadded).sum(dim=(1, 2)) / count
            centered = (patched - loc[:, None, None]) * unpadded
            scale = (centered.square().sum(dim=(1, 2)) / count).clamp_min(0.0).sqrt()
            return loc, scale.clamp_min(self.tolerance)

        # first_patch: the first patch with more than three unpadded values,
        # falling back to the last patch when no patch qualifies.
        unpadded_per_patch = (1.0 - pads).sum(dim=2)
        qualifies = (unpadded_per_patch >= 3).to(torch.int32)
        indices = torch.argmax(qualifies, dim=1)
        patch_index = torch.where(
            qualifies.sum(dim=1) == 0, patched.shape[1] - 1, indices
        )
        rows = torch.arange(batch_size, device=source.device)
        selected = patched[rows, patch_index, :]
        mask = 1.0 - pads[rows, patch_index, :]

        count = mask.sum(dim=1).clamp_min(1.0)
        loc = (selected * mask).sum(dim=1) / count
        centered = (selected - loc.unsqueeze(-1)) * mask
        variance = (centered.square().sum(dim=1) / count).clamp_min(0.0)
        return loc, variance.sqrt().clamp_min(self.tolerance)

    def forward(
        self, x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        normalized = (x - loc) / scale
        return torch.where(
            torch.abs(x - self.pad_val) < self.tolerance,
            torch.tensor(self.pad_val, dtype=normalized.dtype, device=x.device),
            normalized,
        )


class TimesFMNormalization(normalization.BackboneNormalization):
    """TimesFM's instance norm, reproduced exactly.

    `disable` rebinds two methods rather than swapping a submodule, because
    TimesFM's normalization is `_forward_transform`/`_reverse_transform` on the
    decoder itself and there is no attribute to replace. Binding on the
    instance leaves the vendored class untouched.
    """

    normalized_condition = "timesfm_normalized"
    original_condition = "timesfm_native_original"

    def __init__(self, config: TimesFMConfig, mode: str = "first_patch"):
        super().__init__(
            TimesFMScheme(
                mode=mode,
                patch_len=config.patch_len,
                tolerance=config.tolerance,
                pad_val=config.pad_val,
            )
        )

    def disable(self, model: PatchedTimeSeriesDecoder) -> None:
        def _forward_transform(self, inputs, patched_pads):
            mu = torch.zeros(inputs.shape[0], dtype=inputs.dtype, device=inputs.device)
            sigma = torch.ones(
                inputs.shape[0], dtype=inputs.dtype, device=inputs.device
            )
            return inputs, (mu, sigma)

        def _reverse_transform(self, outputs, stats):
            return outputs

        model._forward_transform = types.MethodType(_forward_transform, model)
        model._reverse_transform = types.MethodType(_reverse_transform, model)


CONDITIONS = TimesFMNormalization.conditions()

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


def frequency_bucket(freq: str) -> int:
    """Maps a pandas-style frequency string to TimesFM's 3-way frequency
    embedding bucket (0 = sub-daily, 1 = daily/weekly, 2 = monthly or coarser).

    This is our own bucketing for training the freq_emb from scratch on
    GiftEvalPretrain; it is not claimed to reproduce Google's original
    checkpoint's bucket semantics, which don't matter here since freq_emb is
    randomly initialized and learned fresh.
    """
    offset = seasonality.parse_offset(freq)
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
            std = row["context_std"]
            context[context_valid] = b * (context[context_valid] - mean) / std
            context[~context_valid] = 0.0
            target[tgt_valid] = b * (target[tgt_valid] - mean) / std
            target[~tgt_valid] = 0.0
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


def _preprocess_whole_context(
    model: PatchedTimeSeriesDecoder, batch: TimesFMBatch
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """TimesFM preprocessing with causal statistics from the whole context."""
    batch_size = batch.context.shape[0]
    patched_inputs = batch.context.view(batch_size, -1, model.config.patch_len)
    patched_pads = batch.context_padding.view(batch_size, -1, model.config.patch_len)
    patched_inputs = torch.where(patched_pads == 1.0, 0.0, patched_inputs)
    patched_pads = torch.where(
        torch.abs(patched_inputs - model.config.pad_val) < model.config.tolerance,
        torch.ones_like(patched_pads),
        patched_pads,
    )

    valid = 1.0 - patched_pads
    count = valid.sum(dim=(1, 2)).clamp_min(1.0)
    mu = (patched_inputs * valid).sum(dim=(1, 2)) / count
    centered = (patched_inputs - mu[:, None, None]) * valid
    sigma = torch.sqrt((centered.square().sum(dim=(1, 2)) / count).clamp_min(0.0))
    sigma = sigma.clamp_min(model.config.tolerance)
    normalized = centered / sigma[:, None, None]

    concat_inputs = torch.cat([normalized, patched_pads], dim=-1)
    model_input = model.input_ff_layer(concat_inputs)
    patched_padding = torch.min(patched_pads, dim=-1)[0]
    if model.config.use_positional_embedding:
        pos_emb = model.position_emb(model_input.shape[1]).to(model_input.device)
        pos_emb = torch.concat([pos_emb] * model_input.shape[0], dim=0)
        model_input += _shift_padded_seq(patched_padding, pos_emb)
    return model_input, patched_padding, (mu, sigma)


def run_decoder(
    model: PatchedTimeSeriesDecoder,
    batch: TimesFMBatch,
    normalization_mode: str = "first_patch",
):
    """Forward pass exposing both the normalized (pre-inverse-transform) and
    original-space (post-inverse-transform) last-patch output, following the
    exact same submodule sequence as `PatchedTimeSeriesDecoder.forward`."""
    num_outputs = len(model.config.quantiles) + 1
    if normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(f"normalization_mode must be one of {NORMALIZATION_MODES}")
    if normalization_mode == "first_patch":
        model_input, patched_padding, stats, _ = model._preprocess_input(
            input_ts=batch.context, input_padding=batch.context_padding
        )
    else:
        model_input, patched_padding, stats = _preprocess_whole_context(model, batch)
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
    mse_per_example: torch.Tensor  # [B], in `condition`'s space (the objective)
    pinball_per_example: torch.Tensor  # [B]
    loss_per_example: torch.Tensor  # [B], mse + mean(pinball), in `condition`'s space
    normalized_mse: torch.Tensor  # [B], always in normalized space
    mase: torch.Tensor  # [B], original-space MAE / seasonal-naive MAE
    degenerate: torch.Tensor  # [B] bool, sigma sitting on the clamp floor
    original_point_forecast: torch.Tensor  # [B, horizon_len]


def forward(
    model: PatchedTimeSeriesDecoder,
    batch: TimesFMBatch,
    condition: str,
    normalization_mode: str = "first_patch",
) -> TimesFMForwardResult:
    if condition not in CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r}, must be one of {CONDITIONS}"
        )

    normalized_out, original_out, (mu, sigma) = run_decoder(
        model, batch, normalization_mode
    )
    quantiles = model.config.quantiles

    # Eligibility must not depend on the imposed scale. Dividing the selected
    # normalization sigma by the known b recovers its pre-intervention value.
    # This also catches a near-flat first patch that crosses the 1e-6 clamp only
    # after receiving b=10, which the previous post-intervention check retained
    # in one assignment and dropped in its complement.
    base_sigma = sigma / batch.scale
    degenerate = base_sigma <= model.config.tolerance * (1.0 + 1e-6)

    if condition == "timesfm_native_original":
        point = original_out[..., 0]
        quantile_pred = original_out[..., 1:]
        target = batch.target
    else:
        point = normalized_out[..., 0]
        quantile_pred = normalized_out[..., 1:]
        target = (batch.target - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)

    mse = pointwise.masked_mse(point, target, batch.target_valid, reduction="none")
    pinball = quantile.pinball_loss(
        quantile_pred,
        target,
        quantiles,
        valid=batch.target_valid,
        reduction="none",
    )

    # Evaluation metrics, computed regardless of which space `condition`
    # optimizes in so both arms are scored on identical definitions.
    normalized_target = (batch.target - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
    normalized_mse = pointwise.masked_mse(
        normalized_out[..., 0], normalized_target, batch.target_valid, reduction="none"
    )
    original_mae = pointwise.masked_mae(
        original_out[..., 0], batch.target, batch.target_valid, reduction="none"
    )
    periods = torch.as_tensor(
        [seasonality.seasonal_period(f) for f in batch.frequency],
        device=batch.target.device,
    )
    # context_padding is 1 = missing (TimesFM convention), so validity is its
    # complement.
    naive_mae = forecast.seasonal_naive_mae(
        batch.context, 1.0 - batch.context_padding, periods
    )

    # Degenerate windows carry no usable normalized-space target, so they are
    # excluded from every reported metric (NaN is dropped downstream by
    # group_mean_by_source / pooled_mean) rather than silently dominating them.
    nan = torch.tensor(float("nan"), device=normalized_mse.device)
    return TimesFMForwardResult(
        mse_per_example=mse,
        pinball_per_example=pinball,
        loss_per_example=mse + pinball,
        normalized_mse=torch.where(degenerate, nan, normalized_mse),
        mase=torch.where(degenerate, nan, original_mae / naive_mae),
        degenerate=degenerate,
        original_point_forecast=original_out[..., 0],
    )


def training_step_metrics(
    model: PatchedTimeSeriesDecoder,
    batch: TimesFMBatch,
    condition: str,
    normalization_mode: str,
    objective: str,
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float,
) -> dict:
    optimizer.zero_grad(set_to_none=True)
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    result = forward(model, batch, condition, normalization_mode)
    # Windows whose pre-intervention normalization sigma sits on the clamp
    # floor have no well-defined normalized-space target.
    keep = ~result.degenerate
    if not bool(keep.any()):
        raise RuntimeError("every window in the batch has a degenerate sigma")
    objective_loss = (
        result.mse_per_example if objective == "mse" else result.loss_per_example
    )
    loss = objective_loss[keep].mean()
    output_head_params = list(model.horizon_ff_layer.parameters())
    gradient_metrics = gradients.backward_with_safe_gradient_clipping(
        loss,
        model.parameters(),
        grad_clip_norm,
        tracked_parameters=output_head_params,
    )
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "mse_per_example": result.mse_per_example.detach().cpu().numpy(),
        "pinball_per_example": result.pinball_per_example.detach().cpu().numpy(),
        "mse_to_pinball_loss_ratio": float(
            result.mse_per_example.detach().mean()
            / result.pinball_per_example.detach().mean().clamp_min(1e-12)
        ),
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
class TimesFMForecaster:
    """`src.eval.protocol.Forecaster` over a trained TimesFM checkpoint.

    Scoring reads the quantile heads in original units, which is the space
    every suite's actuals live in, regardless of which space the run's
    `condition` optimized. That mirrors how `forward` already computes its
    eval metrics on fixed definitions so both arms stay comparable.
    """

    model: PatchedTimeSeriesDecoder
    device: str
    normalization_mode: str
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
        buckets = np.array([frequency_bucket(f) for f in freqs], dtype=np.int64)
        batch_size = context.shape[0]
        # run_decoder reads only the tensor fields; the numpy columns exist
        # for the training path's per-source breakdowns and are unused here.
        batch = TimesFMBatch(
            context=torch.from_numpy(context.astype(np.float32)),
            context_padding=torch.from_numpy((1.0 - valid).astype(np.float32)),
            target=torch.zeros(batch_size, self.horizon),
            target_valid=torch.zeros(batch_size, self.horizon),
            freq=torch.from_numpy(buckets).unsqueeze(1),
            dataset=np.empty(batch_size, dtype=object),
            domain=np.empty(batch_size, dtype=object),
            frequency=np.asarray(freqs, dtype=object),
            scale=torch.ones(batch_size),
        ).to(self.device)

        with torch.no_grad():
            _, original_out, _ = run_decoder(self.model, batch, self.normalization_mode)
        # Column 0 is the point head, the rest are the quantile heads.
        return original_out[..., 1:].float().cpu().numpy()


def build_forecaster(cfg, checkpoint_path, device: str) -> TimesFMForecaster:
    """Loads a trained TimesFM checkpoint for evaluation.

    The eval context length comes from the run's own
    `window_index.context_length` so the model sees the shape it trained on.
    """
    from src.training.tsfm import TIMESFM_CONFIGS

    model_config = TIMESFM_CONFIGS[cfg.timesfm.config_size]
    model = build_timesfm_model(model_config, seed=cfg.seed)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device).eval()

    context_length = int(cfg.window_index.context_length)
    if context_length % model_config.patch_len != 0:
        raise ValueError(
            f"context_length {context_length} is not a multiple of patch_len "
            f"{model_config.patch_len}, which the patching in _preprocess_input "
            "requires"
        )
    return TimesFMForecaster(
        model=model,
        device=device,
        normalization_mode=cfg.timesfm.normalization_mode,
        context_length=context_length,
        horizon=model_config.horizon_len,
        quantiles=list(model_config.quantiles),
    )


from src.data import seasonality
from src.losses import pointwise, quantile
from src.metrics import forecast
from src.training import gradients

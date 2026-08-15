"""Normalization as a swappable module rather than a backbone internal.

Every forecaster in this repo normalizes its input with per-window statistics
and then has to decide which space the loss is computed in. That decision was
reimplemented in each adapter. Here it is one object.

A `NormalizationModule` exposes two calls. `transform_input` derives the
statistics and normalizes the context, returning a `TransformStats` that the
caller threads forward. `transform_target_and_output` puts the model output and
the target into a common space so the loss can be taken. `SIT` moves the target
into normalized space, `RevIN` moves the output back into original space.
Neither touches the loss itself.

The statistics are returned rather than stored on the module so that gradient
accumulation, evaluation interleaved with training, and DDP cannot cross
contaminate.

Schemes are registered by name because they are not cosmetically different:
epsilon placement, Bessel correction, NaN handling, and (for Chronos-2) a
nonlinearity all vary between upstream implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransformStats:
    """Per-window statistics plus the scheme that produced them.

    `loc` and `scale` are [B]. `forward` and `inverse` broadcast them against
    any input whose leading dimension is the batch, so a [B, T] target and a
    [B, Q, T] quantile prediction both work.
    """

    loc: torch.Tensor
    scale: torch.Tensor
    degenerate: torch.Tensor
    scheme: "NormalizationScheme"

    def _broadcast(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return self.loc.view(shape), self.scale.view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loc, scale = self._broadcast(x)
        return self.scheme.forward(x, loc, scale)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        loc, scale = self._broadcast(z)
        return self.scheme.inverse(z, loc, scale)


class NormalizationScheme(ABC):
    """How statistics are derived and how the transform is applied.

    `scale_floor` is the smallest scale the scheme can report. A window sitting
    on it is constant to within numerical tolerance and carries no usable
    normalized-space target, which callers flag as degenerate.
    """

    scale_floor: float

    @abstractmethod
    def statistics(
        self, source: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (loc, scale), both [B], from a [B, T] source and mask."""

    def forward(
        self, x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return (x - loc) / scale

    def inverse(
        self, z: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return z * scale + loc


class StandardScheme(NormalizationScheme):
    """Mean and standard deviation over valid positions, epsilon added after.

    Matches `PatchTransformer.normalize`.
    """

    def __init__(self, eps: float = 1e-5, correction: int = 1):
        self.eps = eps
        self.correction = correction
        self.scale_floor = eps

    def statistics(
        self, source: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = valid.sum(dim=-1).clamp_min(1.0)
        loc = (source * valid).sum(dim=-1) / count
        centered = (source - loc.unsqueeze(-1)) * valid
        denominator = (count - self.correction).clamp_min(1.0)
        scale = (centered.square().sum(dim=-1) / denominator).sqrt() + self.eps
        return loc, scale


class MomentRevINScheme(NormalizationScheme):
    """MOMENT's RevIN: NaN-masked mean and uncorrected std, epsilon added after.

    `vendor/moment/models/layers/revin.py` masks invalid positions to NaN and
    reduces with nanmean, which is the uncorrected (population) variance.
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.scale_floor = eps

    def statistics(
        self, source: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked = torch.where(valid.bool(), source, torch.nan)
        loc = masked.nanmean(dim=-1)
        centered = masked - loc.unsqueeze(-1)
        scale = centered.square().nanmean(dim=-1).sqrt() + self.eps
        return loc, scale


class Chronos2Scheme(NormalizationScheme):
    """Chronos-2's InstanceNorm: standardize, then optionally arcsinh.

    The arcsinh is not a loc/scale operation, which is why this scheme
    overrides `forward` and `inverse` rather than reusing the affine default.
    It is invertible (`sinh` is exact) but not affine, so this model's
    normalized and original spaces are related nonlinearly where the other
    three are related by an exact affine map.

    `inverse` casts to float32 before `sinh`, matching upstream, because sinh
    overflows quickly in reduced precision.
    """

    def __init__(self, eps: float = 1e-5, use_arcsinh: bool = True):
        self.eps = eps
        self.use_arcsinh = use_arcsinh
        self.scale_floor = eps

    def statistics(
        self, source: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.where(valid.bool(), source, torch.nan).to(torch.float32)
        loc = torch.nan_to_num(x.nanmean(dim=-1), nan=0.0)
        scale = torch.nan_to_num(
            (x - loc.unsqueeze(-1)).square().nanmean(dim=-1).sqrt(), nan=1.0
        )
        scale = torch.where(scale == 0, torch.tensor(self.eps, device=x.device), scale)
        return loc, scale

    def forward(
        self, x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        original_dtype = x.dtype
        scaled = (x.to(torch.float32) - loc) / scale
        if self.use_arcsinh:
            scaled = torch.arcsinh(scaled)
        return scaled.to(original_dtype)

    def inverse(
        self, z: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        original_dtype = z.dtype
        z = z.to(torch.float32)
        if self.use_arcsinh:
            z = torch.sinh(z)
        return (z * scale + loc).to(original_dtype)


class Moirai2StdScheme(NormalizationScheme):
    """uni2ts `PackedStdScaler`: Bessel-corrected variance, floor under the sqrt.

    Upstream adds `minimum_scale` to the variance before taking the square
    root rather than to the resulting scale, so the smallest reportable scale
    is its square root.
    """

    def __init__(self, correction: int = 1, minimum_scale: float = 1e-5):
        self.correction = correction
        self.minimum_scale = minimum_scale
        self.scale_floor = minimum_scale**0.5

    def statistics(
        self, source: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = source.double()
        mask = valid.double()
        count = mask.sum(dim=-1)
        loc = torch.where(count > 0, (x * mask).sum(dim=-1) / count.clamp_min(1.0), 0.0)
        centered = (x - loc.unsqueeze(-1)).square() * mask
        denominator = count - self.correction
        variance = torch.where(
            denominator > 0, centered.sum(dim=-1) / denominator.clamp_min(1.0), 0.0
        )
        scale = (variance + self.minimum_scale).sqrt()
        return loc.float(), scale.float()


class Moirai2AbsMeanScheme(NormalizationScheme):
    """uni2ts `PackedAbsMeanScaler`: zero location, mean absolute value scale."""

    def __init__(self):
        self.scale_floor = 0.0

    def statistics(
        self, source: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = source.double()
        mask = valid.double()
        count = mask.sum(dim=-1)
        scale = torch.where(
            count > 0, (x.abs() * mask).sum(dim=-1) / count.clamp_min(1.0), 0.0
        )
        return torch.zeros_like(scale).float(), scale.float()


class TimesFMScheme(NormalizationScheme):
    """TimesFM's instance norm, in either of the two modes this repo uses.

    `first_patch` reproduces vendored `_masked_mean_std`, which takes the
    statistics of the first patch holding more than three unpadded values.
    `whole_context` takes them over the entire valid context instead, which is
    this repo's causal variant (see `timesfm._preprocess_whole_context`).

    `forward` reproduces `_forward_transform`'s pad_val overwrite, which resets
    positions that were already the padding sentinel back to it after scaling.
    """

    MODES = ("first_patch", "whole_context")

    def __init__(
        self,
        mode: str = "first_patch",
        patch_len: int = 32,
        tolerance: float = 1e-6,
        pad_val: float = 1123581321.0,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
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


SCHEMES = {
    "std": StandardScheme,
    "moment_revin": MomentRevINScheme,
    "chronos2": Chronos2Scheme,
    "moirai2_std": Moirai2StdScheme,
    "moirai2_absmean": Moirai2AbsMeanScheme,
    "timesfm": TimesFMScheme,
}


def build_scheme(name: str, **kwargs) -> NormalizationScheme:
    if name not in SCHEMES:
        raise ValueError(f"unknown scheme {name!r}, must be one of {tuple(SCHEMES)}")
    return SCHEMES[name](**kwargs)


class NormalizationModule(nn.Module, ABC):
    """Owns the normalization so the backbone and the loss do not.

    Wrap a backbone whose own normalization has been disabled (see
    `src/models/norm_shims.py`) and the three-line loop is the same for every
    model:

        z_context, stats = norm.transform_input(context, valid)
        output = backbone(z_context)
        output, target = norm.transform_target_and_output(output, target, stats)
    """

    def __init__(self, scheme: NormalizationScheme, apply_causal_norm: bool = False):
        super().__init__()
        self.scheme = scheme
        self.apply_causal_norm = apply_causal_norm

    def transform_input(
        self,
        context: torch.Tensor,
        valid: torch.Tensor | None = None,
        extra_context: torch.Tensor | None = None,
        extra_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TransformStats]:
        """Normalizes the context and returns the statistics used to do it.

        With `apply_causal_norm`, the statistics come from `extra_context`, a
        strictly earlier window, so the context itself never enters them.
        Without it, they come from the context.
        """
        if self.apply_causal_norm and extra_context is None:
            raise ValueError(
                "apply_causal_norm is set but extra_context is None, so there "
                "is no prior window to take statistics from"
            )

        if valid is None:
            valid = torch.ones_like(context)
        if self.apply_causal_norm:
            source = extra_context
            source_valid = (
                extra_valid if extra_valid is not None else torch.ones_like(source)
            )
        else:
            source, source_valid = context, valid

        loc, scale = self.scheme.statistics(source, source_valid)
        stats = TransformStats(
            loc=loc,
            scale=scale,
            degenerate=scale <= self.scheme.scale_floor * (1.0 + 1e-6),
            scheme=self.scheme,
        )
        return stats.forward(context), stats

    @abstractmethod
    def transform_target_and_output(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        stats: TransformStats,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Puts output and target into a common space for the loss."""

    def align_target_output(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        stats: TransformStats,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform_target_and_output(output, target, stats)


class SIT(NormalizationModule):
    """Scale-invariant training: the loss is taken in normalized space.

    The output is already normalized, so only the target moves.
    """

    def transform_target_and_output(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        stats: TransformStats,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return output, stats.forward(target)


class RevIN(NormalizationModule):
    """Reversible instance norm: the loss is taken in original space.

    The target is already in original space, so only the output moves.
    """

    def transform_target_and_output(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        stats: TransformStats,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return stats.inverse(output), target

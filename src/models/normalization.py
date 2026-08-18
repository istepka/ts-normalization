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

Nothing here knows about any particular backbone. The schemes are named for the
statistic they compute, and each adapter says which one it needs by subclassing
`BackboneNormalization` in its own module.
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

    The default `forward`/`inverse` are affine. A scheme whose transform is not
    affine overrides both, and `TransformStats` routes through the scheme
    rather than through a bare `(loc, scale)` pair so that callers do not have
    to know which kind they hold.
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
    """Mean and Bessel-corrected std over valid positions, epsilon added after.

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


class PopulationStdScheme(NormalizationScheme):
    """Uncorrected std over valid positions via NaN reduction, epsilon after.

    Kept separate from `StandardScheme(correction=0)` rather than folded into
    it because the NaN reduction is a real behavioral difference and not just
    an arithmetic one. A window with no valid positions yields NaN statistics
    here and clamped ones there.

    This is what MOMENT's RevIN computes, pinned in
    `tests/test_moment_adapter.py`.
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


class FlooredStdScheme(NormalizationScheme):
    """Bessel-corrected std with the floor added under the square root.

    Adding the floor to the variance rather than to the resulting scale makes
    the smallest reportable scale its square root, which is why this cannot be
    expressed as `StandardScheme` with a different epsilon.

    This is what uni2ts's `PackedStdScaler` computes, pinned in
    `tests/test_moirai2_adapter.py`.
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


class AbsMeanScheme(NormalizationScheme):
    """Zero location, mean absolute value scale.

    This is what uni2ts's `PackedAbsMeanScaler` computes, pinned in
    `tests/test_moirai2_adapter.py`.
    """

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


class ArcsinhStdScheme(NormalizationScheme):
    """Standardize, then arcsinh. Invertible but not affine.

    The arcsinh is not a loc/scale operation, so this scheme overrides
    `forward` and `inverse` instead of reusing the affine default. It is fully
    reversible, `sinh` being its exact inverse, but the normalized and original
    spaces are related nonlinearly rather than by a per-window constant. That
    matters when comparing loss spaces, so it belongs in the paper's model
    table.

    `inverse` casts to float32 before `sinh` because sinh overflows quickly in
    reduced precision.

    This is what Chronos-2's `InstanceNorm` computes, pinned in
    `tests/test_chronos2_adapter.py`.
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


class NormalizationModule(nn.Module, ABC):
    """Owns the normalization so the backbone and the loss do not.

    Wrap a backbone whose own normalization has been disabled and the loop is
    the same for every model:

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


class BackboneNormalization(ABC):
    """What one backbone contributes to the loss-space contrast.

    Adding a baseline means subclassing this in that model's adapter module
    and supplying three things:

    1. `scheme`, the statistic the backbone normalizes with, either one from
       this module or a subclass of `NormalizationScheme` if the backbone's
       transform is entangled with its own data encoding.
    2. `normalized_condition` and `original_condition`, the two condition
       names the training config selects between. They are named rather than
       positional because the existing adapters do not agree on an order.
    3. `disable`, which makes the backbone's built-in normalization the
       identity so it does not normalize a second time.

    `disable` lives here rather than in a shared module because it is the one
    genuinely backbone-specific piece. It reaches into a particular attribute
    or method of a particular vendored class, so it belongs next to the rest of
    that model's quirks.
    """

    normalized_condition: str
    original_condition: str

    def __init__(self, scheme: NormalizationScheme, apply_causal_norm: bool = False):
        self.scheme = scheme
        self.apply_causal_norm = apply_causal_norm
        # `SIT` and `RevIN` inherit the same `transform_input`, so either one
        # serves the input half. See `transform_input` below.
        self._input = SIT(scheme, apply_causal_norm)

    def transform_input(
        self,
        context: torch.Tensor,
        valid: torch.Tensor | None = None,
        extra_context: torch.Tensor | None = None,
        extra_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TransformStats]:
        """Normalizes the context, without picking a condition.

        The condition changes only which space the loss is read in, never the
        forward pass, so a caller that needs the normalized context and its
        statistics does not have to choose one first. That is what makes the
        two conditions comparable: they are the same forward pass.
        """
        return self._input.transform_input(context, valid, extra_context, extra_valid)

    @classmethod
    def conditions(cls) -> tuple[str, str]:
        """The two condition names, readable without building an instance.

        A classmethod because most schemes need the model config to construct,
        while the training loop validates `cfg.condition` before it has one.
        """
        return (cls.normalized_condition, cls.original_condition)

    @abstractmethod
    def disable(self, model) -> None:
        """Turns the backbone's built-in normalization into the identity.

        Must not edit vendored source. Swapping a submodule or rebinding a
        method on the instance keeps the pinned upstream files byte-identical
        to their REVISION.
        """

    def module(
        self, condition: str, apply_causal_norm: bool = False
    ) -> NormalizationModule:
        """Returns the `SIT` or `RevIN` this condition asks for."""
        if condition == self.normalized_condition:
            return SIT(self.scheme, self.apply_causal_norm or apply_causal_norm)
        if condition == self.original_condition:
            return RevIN(self.scheme, self.apply_causal_norm or apply_causal_norm)
        raise ValueError(
            f"unknown condition {condition!r}, must be one of {self.conditions()}"
        )

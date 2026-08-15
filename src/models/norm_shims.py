"""Turns each backbone's built-in normalization into the identity.

Once `src/models/normalization.py` owns the transform, the backbone must not
normalize a second time. None of these edit vendored source: three swap an
attribute and one rebinds two methods on the instance, so the pinned upstream
files stay byte-identical to their REVISION.

Each shim keeps whatever attributes the adapters read off the normalizer it
replaces (`eps`, `minimum_scale`, `mean`, `stdev`), so disabling normalization
does not require touching the surrounding metric code.
"""

import types

import torch
from torch import nn


class IdentityRevIN(nn.Module):
    """Stands in for MOMENT's `RevIN` with the same call signature.

    `mean` and `stdev` are exposed because `moment.forward` reads them out of
    the model's metadata to build its normalized-space target.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine = False
        self.mean = torch.zeros(1)
        self.stdev = torch.ones(1)

    def forward(self, x: torch.Tensor, mode: str = "norm", mask=None):
        if mode not in ("norm", "denorm"):
            raise ValueError(f"mode must be 'norm' or 'denorm', got {mode!r}")
        if mode == "norm":
            self.mean = torch.zeros(
                x.shape[0], x.shape[1], 1, dtype=x.dtype, device=x.device
            )
            self.stdev = torch.ones(
                x.shape[0], x.shape[1], 1, dtype=x.dtype, device=x.device
            )
        return x


class IdentityInstanceNorm(nn.Module):
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


def disable_moment_normalization(model) -> None:
    model.normalizer = IdentityRevIN(eps=model.normalizer.eps)


def disable_chronos2_normalization(model) -> None:
    model.instance_norm = IdentityInstanceNorm(eps=model.instance_norm.eps)


def disable_moirai2_normalization(model) -> None:
    """Swaps in uni2ts's own `PackedNOPScaler`, which returns loc=0, scale=1.

    `minimum_scale` is carried over because the adapter derives its degenerate
    floor from it.
    """
    from src.models.vendor.moirai2.packed_scaler import PackedNOPScaler

    minimum_scale = model.scaler.minimum_scale
    model.scaler = PackedNOPScaler()
    model.scaler.minimum_scale = minimum_scale


def disable_timesfm_normalization(model) -> None:
    """Rebinds `_forward_transform` and `_reverse_transform` to identity.

    These are methods rather than a submodule, so there is no attribute to
    swap. Binding on the instance leaves the vendored class untouched.
    """

    def _forward_transform(self, inputs, patched_pads):
        mu = torch.zeros(inputs.shape[0], dtype=inputs.dtype, device=inputs.device)
        sigma = torch.ones(inputs.shape[0], dtype=inputs.dtype, device=inputs.device)
        return inputs, (mu, sigma)

    def _reverse_transform(self, outputs, stats):
        return outputs

    model._forward_transform = types.MethodType(_forward_transform, model)
    model._reverse_transform = types.MethodType(_reverse_transform, model)

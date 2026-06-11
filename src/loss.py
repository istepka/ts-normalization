"""Loss objectives in normalized vs. original space.

Both share the same forward pass (z_pred, a, b); the only difference is the space
in which the MSE is taken. For p=2 the original-space loss equals a b^2-weighted
normalized-space loss, which is exactly the gradient scaling under analysis.
"""

import torch


def normalize_target(target: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    return (target - a) / b


def compute_loss(
    mode: str,
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    if mode == "normalized":
        residual = z_pred - z_target
    elif mode == "original":
        residual = b * (z_pred - z_target)
    else:
        raise ValueError(f"unknown loss mode: {mode}")
    return residual.pow(2).mean()


def per_sample_nmse(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """Normalized-space MSE per sample; the common metric for both runs."""
    return (z_pred - z_target).pow(2).mean(dim=1)

"""Gradient-norm bookkeeping shared by the training loops."""

import numpy as np
import torch


def grad_norm(parameters) -> float:
    """L2 norm of gradients across `parameters` (call before/after clipping)."""
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm(2) ** 2)
    return total**0.5


def backward_with_safe_gradient_clipping(
    loss: torch.Tensor,
    parameters,
    max_norm: float,
    tracked_parameters=(),
) -> dict[str, float | bool]:
    """Backpropagate and clip without overflowing the gradient-norm reduction.

    A temporary scalar multiplier keeps the backward pass representable. The
    multiplier is removed analytically when computing the original gradient
    norm and final clipping coefficient, so the optimizer receives the same
    clipped gradient direction and magnitude as an exact unscaled reduction.
    """
    parameters = list(parameters)
    tracked_parameters = list(tracked_parameters)
    loss_value = float(loss.detach().abs())
    if not np.isfinite(loss_value):
        raise RuntimeError("training loss is not finite")

    backward_scale = 1.0 / max(1.0, loss_value)
    (loss * backward_scale).backward()
    scaled_norm = grad_norm(parameters)
    scaled_tracked_norm = grad_norm(tracked_parameters)
    if not np.isfinite(scaled_norm) or not np.isfinite(scaled_tracked_norm):
        raise RuntimeError("scaled gradient norm is not finite")

    norm_before_clip = scaled_norm / backward_scale
    tracked_norm_before_clip = scaled_tracked_norm / backward_scale
    clip_coefficient = min(1.0, max_norm / (norm_before_clip + 1e-6))
    gradient_multiplier = clip_coefficient / backward_scale
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(gradient_multiplier)

    return {
        "tracked_norm_before_clip": tracked_norm_before_clip,
        "total_norm_before_clip": norm_before_clip,
        "total_norm_after_clip": grad_norm(parameters),
        "clipped": clip_coefficient < 1.0,
    }

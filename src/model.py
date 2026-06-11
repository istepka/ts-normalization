"""Tiny univariate patch-based transformer with RevIN-style instance norm.

The model consumes the context window, normalizes it with per-instance statistics
(a = mean, b = std of the context), and predicts the horizon in normalized space.
The instance statistics (a, b) are returned so the trainer can compute loss in
either the normalized space (on z) or the original space (on b * z + a).
"""

import torch
from omegaconf import DictConfig
from torch import nn


class PatchTransformer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.context_length = cfg.data.context_length
        self.horizon = cfg.data.horizon
        self.patch_length = cfg.model.patch_length
        self.norm_eps = cfg.model.norm_eps

        if self.context_length % self.patch_length != 0:
            raise ValueError(
                f"context_length {self.context_length} must be divisible by "
                f"patch_length {self.patch_length}"
            )
        num_patches = self.context_length // self.patch_length
        d_model = cfg.model.d_model

        self.patch_embed = nn.Linear(self.patch_length, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.model.n_heads,
            dim_feedforward=cfg.model.dim_feedforward,
            dropout=cfg.model.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.model.n_layers)
        self.head = nn.Linear(num_patches * d_model, self.horizon)

    def normalize(self, context: torch.Tensor) -> tuple[torch.Tensor, ...]:
        a = context.mean(dim=1, keepdim=True)
        b = context.std(dim=1, keepdim=True) + self.norm_eps
        z_context = (context - a) / b
        return z_context, a, b

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, ...]:
        z_context, a, b = self.normalize(context)
        patches = z_context.unfold(1, self.patch_length, self.patch_length)
        tokens = self.patch_embed(patches) + self.pos_embed
        encoded = self.encoder(tokens)
        z_pred = self.head(encoded.flatten(1))
        return z_pred, a, b

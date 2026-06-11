"""Synthetic time series for the loss-space comparison toy.

Categories are *distinct* learnable patterns (different frequency and phase) that
share the same mean but live on different variance scales (variance is purely
amplitude-driven). After instance normalization they remain distinguishable, so a
shared model must allocate capacity to each, which is what lets the original-space
loss expose disparate per-category learning under heterogeneous variance.
"""

from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig

VAL_SEED = 12345  # fixed -> the held-out validation set is identical across all runs


@dataclass
class Batch:
    context: torch.Tensor  # [B, context_length], original space
    target: torch.Tensor  # [B, horizon], original space
    category: torch.Tensor  # [B], long category index


class SyntheticTSDataset:
    """Per-category sine series with sliding context->horizon windows.

    Builds a fixed, shared stratified batch schedule once so that every training
    run sees identical samples in identical order; the only thing allowed to differ
    between runs is the loss space.
    """

    def __init__(self, cfg: DictConfig, generator: torch.Generator):
        self.context_length = cfg.data.context_length
        self.horizon = cfg.data.horizon
        self.window_length = self.context_length + self.horizon
        self.category_names = [c.name for c in cfg.data.categories]
        self.num_categories = len(cfg.data.categories)

        self.windows = self._build_windows(cfg, train=True)  # [n_windows, win_len] each
        self.batch_schedule = self._build_schedule(cfg, generator)
        self._build_val_set(cfg)

    def _build_windows(self, cfg: DictConfig, *, train: bool) -> list[torch.Tensor]:
        """Per-category sliding windows. Training series sit at phases
        `s * 2pi / series_per_category`; validation series add a half-step offset
        (`pi / series_per_category`) so their phases never coincide with training,
        giving a genuinely held-out (unseen-phase) evaluation set."""
        mean = cfg.data.mean
        cycle_length = cfg.data.cycle_length
        series_length = cfg.data.series_length
        spacing = 2.0 * np.pi / cfg.data.series_per_category
        equal_variance = cfg.data.equal_variance
        n_series = (
            cfg.data.series_per_category if train else cfg.data.val_series_per_category
        )
        offset = 0.0 if train else 0.5 * spacing

        t = np.arange(series_length, dtype=np.float64)
        windows_per_category = []
        for category in cfg.data.categories:
            scale = 1.0 if equal_variance else category.scale
            omega = 2.0 * np.pi * category.freq / cycle_length
            series_windows = []
            for s in range(n_series):
                phase = category.phase + offset + s * spacing
                signal = mean + scale * np.sin(omega * t + phase)
                series_windows.append(self._sliding_windows(signal))
            windows = np.concatenate(series_windows, axis=0)
            windows_per_category.append(torch.from_numpy(windows).float())
        return windows_per_category

    def _build_val_set(self, cfg: DictConfig):
        """Sample a fixed, large pool of held-out windows per category (constant seed,
        so every setup/seed evaluates on the identical set). Stored as flat tensors
        `val_context / val_target / val_category` for a single forward pass per eval."""
        val_windows = self._build_windows(cfg, train=False)
        n = cfg.data.val_windows_per_category
        gen = torch.Generator().manual_seed(VAL_SEED)
        contexts, targets, categories = [], [], []
        for c, windows in enumerate(val_windows):
            pick = torch.randint(len(windows), (n,), generator=gen)
            chosen = windows[pick]
            contexts.append(chosen[:, : self.context_length])
            targets.append(chosen[:, self.context_length :])
            categories.append(torch.full((n,), c, dtype=torch.long))
        self.val_context = torch.cat(contexts, dim=0)
        self.val_target = torch.cat(targets, dim=0)
        self.val_category = torch.cat(categories, dim=0)

    def _sliding_windows(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal) - self.window_length + 1
        idx = np.arange(self.window_length)[None, :] + np.arange(n)[:, None]
        return signal[idx]

    def _build_schedule(
        self, cfg: DictConfig, generator: torch.Generator
    ) -> list[Batch]:
        batch_size = cfg.train.batch_size
        if batch_size % self.num_categories != 0:
            raise ValueError(
                f"batch_size {batch_size} must be divisible by "
                f"num_categories {self.num_categories}"
            )
        per_category = batch_size // self.num_categories
        steps = cfg.train.steps

        schedule = []
        for _ in range(steps):
            contexts, targets, categories = [], [], []
            for c, windows in enumerate(self.windows):
                pick = torch.randint(len(windows), (per_category,), generator=generator)
                chosen = windows[pick]
                contexts.append(chosen[:, : self.context_length])
                targets.append(chosen[:, self.context_length :])
                categories.append(torch.full((per_category,), c, dtype=torch.long))
            schedule.append(
                Batch(
                    context=torch.cat(contexts, dim=0),
                    target=torch.cat(targets, dim=0),
                    category=torch.cat(categories, dim=0),
                )
            )
        return schedule

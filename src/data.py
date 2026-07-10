"""Datasets for the loss-space comparison toy.

The default synthetic dataset uses distinct sine patterns. The real-shape scaled
variant reuses windows from real series, normalizes each base window with its
context statistics, then creates category copies whose only intended difference is
the configured amplitude scale.
"""

from dataclasses import dataclass
from pathlib import Path

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
        self.batch_schedule = build_stratified_schedule(
            self.windows,
            self.context_length,
            cfg.train.batch_size,
            cfg.train.steps,
            generator,
        )
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


class RealShapeScaledDataset:
    """Real window shapes copied across variance categories.

    The input `.npz` must contain the array named by `cfg.data.real_shape_key`.
    Supported shapes:
    - `[N, context+horizon]`: precomputed windows.
    - `[N, T]` or `[T]`: raw series, converted into sliding windows.

    Each base window is normalized using its context mean/std and then rescaled
    by each configured category scale. This keeps the normalized shape identical
    across categories while preserving the same variance intervention as the sine
    experiment.
    """

    def __init__(self, cfg: DictConfig, generator: torch.Generator):
        self.context_length = cfg.data.context_length
        self.horizon = cfg.data.horizon
        self.window_length = self.context_length + self.horizon
        self.category_names = [c.name for c in cfg.data.categories]
        self.num_categories = len(cfg.data.categories)

        base = self._load_base_windows(cfg)
        train_base, val_base = self._split_base_windows(base, cfg)
        self.windows = self._scaled_category_windows(train_base, cfg)
        val_windows = self._scaled_category_windows(val_base, cfg)
        self.batch_schedule = build_stratified_schedule(
            self.windows,
            self.context_length,
            cfg.train.batch_size,
            cfg.train.steps,
            generator,
        )
        self._build_val_set(val_windows, cfg)

    def _load_base_windows(self, cfg: DictConfig) -> torch.Tensor:
        data = np.load(Path(cfg.data.real_shape_path))
        arr = np.asarray(data[cfg.data.real_shape_key], dtype=np.float64)
        if arr.ndim == 1:
            windows = self._sliding_windows(arr)
        elif arr.ndim == 2 and arr.shape[1] == self.window_length:
            windows = arr
        elif arr.ndim == 2 and arr.shape[1] > self.window_length:
            windows = np.concatenate(
                [self._sliding_windows(series) for series in arr], axis=0
            )
        else:
            raise ValueError(
                f"{cfg.data.real_shape_key} must have shape [T], "
                f"[N, T] with T > {self.window_length}, or "
                f"[N, {self.window_length}]"
            )

        finite = np.isfinite(windows).all(axis=1)
        context = windows[:, : self.context_length]
        std = context.std(axis=1)
        keep = finite & (std > 0.0)
        if not np.any(keep):
            raise ValueError("real-shape input has no finite, non-constant windows")
        windows = windows[keep]
        context = windows[:, : self.context_length]
        mean = context.mean(axis=1, keepdims=True)
        std = context.std(axis=1, keepdims=True)
        base = (windows - mean) / std
        return torch.from_numpy(base).float()

    def _split_base_windows(
        self, windows: torch.Tensor, cfg: DictConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(VAL_SEED)
        order = torch.randperm(len(windows), generator=gen)
        n_val = int(len(windows) * cfg.data.real_shape_val_fraction)
        if n_val == 0 or n_val == len(windows):
            raise ValueError("real_shape_val_fraction leaves an empty train/val split")
        val_idx = order[:n_val]
        train_idx = order[n_val:]
        return windows[train_idx], windows[val_idx]

    def _scaled_category_windows(
        self, base_windows: torch.Tensor, cfg: DictConfig
    ) -> list[torch.Tensor]:
        out = []
        for category in cfg.data.categories:
            scale = 1.0 if cfg.data.equal_variance else category.scale
            out.append(cfg.data.mean + scale * base_windows)
        return out

    def _sliding_windows(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal) - self.window_length + 1
        if n <= 0:
            raise ValueError(
                f"series length {len(signal)} must exceed window length "
                f"{self.window_length}"
            )
        idx = np.arange(self.window_length)[None, :] + np.arange(n)[:, None]
        return signal[idx]

    def _build_val_set(self, val_windows: list[torch.Tensor], cfg: DictConfig):
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


def build_stratified_schedule(
    windows_per_category: list[torch.Tensor],
    context_length: int,
    batch_size: int,
    steps: int,
    generator: torch.Generator,
) -> list[Batch]:
    num_categories = len(windows_per_category)
    if batch_size % num_categories != 0:
        raise ValueError(
            f"batch_size {batch_size} must be divisible by "
            f"num_categories {num_categories}"
        )
    per_category = batch_size // num_categories

    schedule = []
    for _ in range(steps):
        contexts, targets, categories = [], [], []
        for c, windows in enumerate(windows_per_category):
            pick = torch.randint(len(windows), (per_category,), generator=generator)
            chosen = windows[pick]
            contexts.append(chosen[:, :context_length])
            targets.append(chosen[:, context_length:])
            categories.append(torch.full((per_category,), c, dtype=torch.long))
        schedule.append(
            Batch(
                context=torch.cat(contexts, dim=0),
                target=torch.cat(targets, dim=0),
                category=torch.cat(categories, dim=0),
            )
        )
    return schedule

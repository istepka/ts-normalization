"""Datasets for the loss-space comparison toy.

The default synthetic dataset uses distinct sine patterns. The real-shape scaled
variant reuses windows from real series, normalizes each base window with its
context statistics, then creates category copies whose only intended difference is
the configured amplitude scale. The real-variance-binned variant groups natural
real windows by context variance. The real scale-swap variant treats datasets as
categories and applies a controlled scale to each dataset after context
normalization.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

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

        train_windows, val_windows = load_real_window_splits(
            cfg, self.window_length, self.context_length
        )
        train_base = torch.from_numpy(
            context_normalize_windows(train_windows, self.context_length, mean=0.0)
        ).float()
        val_base = torch.from_numpy(
            context_normalize_windows(val_windows, self.context_length, mean=0.0)
        ).float()
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

    def _scaled_category_windows(
        self, base_windows: torch.Tensor, cfg: DictConfig
    ) -> list[torch.Tensor]:
        out = []
        for category in cfg.data.categories:
            scale = 1.0 if cfg.data.equal_variance else category.scale
            out.append(cfg.data.mean + scale * base_windows)
        return out

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


class RealVarianceBinnedDataset:
    """Natural real windows grouped by context variance.

    Unlike `RealShapeScaledDataset`, this keeps the naturally scaled real windows
    and creates categories by quantile-binning their context standard deviation.
    The equal-variance control keeps the same bin membership but context-normalizes
    every window to unit scale.
    """

    def __init__(self, cfg: DictConfig, generator: torch.Generator):
        self.context_length = cfg.data.context_length
        self.horizon = cfg.data.horizon
        self.window_length = self.context_length + self.horizon
        self.num_categories = len(cfg.data.categories)
        self.category_names = self._category_names()

        train_windows, val_windows = load_real_window_splits(
            cfg, self.window_length, self.context_length
        )
        thresholds = self._bin_thresholds(train_windows)
        self.windows = self._windows_by_bin(train_windows, thresholds, cfg)
        val_by_bin = self._windows_by_bin(val_windows, thresholds, cfg)
        self.batch_schedule = build_stratified_schedule(
            self.windows,
            self.context_length,
            cfg.train.batch_size,
            cfg.train.steps,
            generator,
        )
        self._build_val_set(val_by_bin, cfg)

    def _category_names(self) -> list[str]:
        if self.num_categories == 3:
            return ["low_var", "mid_var", "high_var"]
        return [f"var_bin_{i}" for i in range(self.num_categories)]

    def _bin_thresholds(self, windows: np.ndarray) -> np.ndarray:
        std = windows[:, : self.context_length].std(axis=1)
        quantiles = np.linspace(0.0, 1.0, self.num_categories + 1)
        return np.quantile(std, quantiles[1:-1])

    def _windows_by_bin(
        self, windows: np.ndarray, thresholds: np.ndarray, cfg: DictConfig
    ) -> list[torch.Tensor]:
        std = windows[:, : self.context_length].std(axis=1)
        assignment = np.searchsorted(thresholds, std, side="right")
        out = []
        for c in range(self.num_categories):
            bin_windows = windows[assignment == c]
            if len(bin_windows) == 0:
                raise ValueError(f"variance bin {c} is empty")
            if cfg.data.equal_variance:
                bin_windows = context_normalize_windows(
                    bin_windows, self.context_length, mean=cfg.data.mean
                )
            out.append(torch.from_numpy(bin_windows).float())
        return out

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


class RealScaleSwapDataset(RealShapeScaledDataset):
    """Eight real datasets with controlled post-normalization scales.

    Each source is split independently with the leakage-free real-window loader.
    Windows are then context-normalized before the configured per-source scale is
    applied. Source ordering is fixed so swapped assignments built from the same
    seed use identical sampled windows and model initialization.
    """

    def __init__(self, cfg: DictConfig, generator: torch.Generator):
        self.context_length = cfg.data.context_length
        self.horizon = cfg.data.horizon
        self.window_length = self.context_length + self.horizon
        self.category_names = [source.name for source in cfg.data.real_sources]
        self.num_categories = len(self.category_names)
        scales = list(cfg.data.scale_assignment)
        if len(scales) != self.num_categories:
            raise ValueError("scale_assignment must contain one scale per real source")

        train_by_source = []
        val_by_source = []
        for source, scale in zip(cfg.data.real_sources, scales):
            source_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            source_cfg.data.real_shape_path = source.path
            source_cfg.data.real_shape_key = source.key
            source_cfg.data.real_value_scale = 1.0
            train_windows, val_windows = load_real_window_splits(
                source_cfg, self.window_length, self.context_length
            )
            train_windows = context_normalize_windows(
                train_windows, self.context_length, mean=0.0
            )
            val_windows = context_normalize_windows(
                val_windows, self.context_length, mean=0.0
            )
            train_by_source.append(torch.from_numpy(scale * train_windows).float())
            val_by_source.append(torch.from_numpy(scale * val_windows).float())

        self.windows = train_by_source
        self.batch_schedule = build_stratified_schedule(
            self.windows,
            self.context_length,
            cfg.train.batch_size,
            cfg.train.steps,
            generator,
        )
        self._build_val_set(val_by_source, cfg)


def load_real_window_splits(
    cfg: DictConfig, window_length: int, context_length: int
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(cfg.data.real_shape_path)) as data:
        arr = np.asarray(data[cfg.data.real_shape_key], dtype=np.float64)

    if arr.ndim == 1:
        train_signal, val_signal = split_contiguous_series(arr, cfg, window_length)
        train_windows = sliding_windows(train_signal, window_length)
        val_windows = sliding_windows(val_signal, window_length)
    elif arr.ndim == 2 and arr.shape[1] == window_length:
        train_windows, val_windows = split_real_rows(arr, cfg)
    elif arr.ndim == 2 and arr.shape[1] > window_length:
        if len(arr) == 1:
            train_signal, val_signal = split_contiguous_series(
                arr[0], cfg, window_length
            )
            train_windows = sliding_windows(train_signal, window_length)
            val_windows = sliding_windows(val_signal, window_length)
        else:
            train_series, val_series = split_real_rows(arr, cfg)
            train_windows = np.concatenate(
                [sliding_windows(series, window_length) for series in train_series],
                axis=0,
            )
            val_windows = np.concatenate(
                [sliding_windows(series, window_length) for series in val_series],
                axis=0,
            )
    else:
        raise ValueError(
            f"{cfg.data.real_shape_key} must have shape [T], "
            f"[N, T] with T > {window_length}, or "
            f"[N, {window_length}]"
        )

    train_windows = filter_real_windows(train_windows, context_length, "training")
    val_windows = filter_real_windows(val_windows, context_length, "validation")
    scale = cfg.data.real_value_scale
    return scale * train_windows, scale * val_windows


def split_real_rows(rows: np.ndarray, cfg: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    gen = torch.Generator().manual_seed(VAL_SEED)
    order = torch.randperm(len(rows), generator=gen).numpy()
    n_val = int(len(rows) * cfg.data.real_shape_val_fraction)
    if n_val == 0 or n_val == len(rows):
        raise ValueError("real_shape_val_fraction leaves an empty train/val split")
    return rows[order[n_val:]], rows[order[:n_val]]


def split_contiguous_series(
    signal: np.ndarray, cfg: DictConfig, window_length: int
) -> tuple[np.ndarray, np.ndarray]:
    n_val = int(len(signal) * cfg.data.real_shape_val_fraction)
    split = len(signal) - n_val
    if split < window_length or n_val < window_length:
        raise ValueError(
            "real_shape_val_fraction leaves fewer than one non-overlapping "
            "window in the training or validation segment"
        )
    return signal[:split], signal[split:]


def filter_real_windows(
    windows: np.ndarray, context_length: int, split_name: str
) -> np.ndarray:
    finite = np.isfinite(windows).all(axis=1)
    std = windows[:, :context_length].std(axis=1)
    keep = finite & (std > 0.0)
    if not np.any(keep):
        raise ValueError(f"{split_name} real input has no finite, non-constant windows")
    return windows[keep]


def sliding_windows(signal: np.ndarray, window_length: int) -> np.ndarray:
    n = len(signal) - window_length + 1
    if n <= 0:
        raise ValueError(
            f"series length {len(signal)} must exceed window length {window_length}"
        )
    idx = np.arange(window_length)[None, :] + np.arange(n)[:, None]
    return signal[idx]


def context_normalize_windows(
    windows: np.ndarray, context_length: int, mean: float
) -> np.ndarray:
    context = windows[:, :context_length]
    context_mean = context.mean(axis=1, keepdims=True)
    context_std = context.std(axis=1, keepdims=True)
    return mean + (windows - context_mean) / context_std


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

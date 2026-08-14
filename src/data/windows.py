"""Window construction and train/val splitting for real series.

Shared by every real-data loss-space variant (shape-scaled, variance-binned,
scale-swap). Everything here is a pure array transform; the dataset classes
that consume it live in `src.data.loss_space`.
"""

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

VAL_SEED = 12345  # fixed -> the held-out validation set is identical across all runs


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

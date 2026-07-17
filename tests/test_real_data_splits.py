import numpy as np
from omegaconf import OmegaConf

from src.data import load_real_window_splits


def real_data_cfg(path, val_fraction: float = 0.2):
    return OmegaConf.create(
        {
            "data": {
                "real_shape_path": str(path),
                "real_shape_key": "data",
                "real_shape_val_fraction": val_fraction,
                "real_value_scale": 1.0,
            }
        }
    )


def test_raw_series_are_split_before_sliding_windows(tmp_path):
    series_ids = np.arange(10, dtype=np.float64)[:, None]
    time = np.arange(100, dtype=np.float64)[None, :]
    series = 1000.0 * series_ids + time
    path = tmp_path / "series.npz"
    np.savez(path, data=series)

    train, val = load_real_window_splits(
        real_data_cfg(path), window_length=80, context_length=64
    )

    train_ids = set(np.floor(train[:, 0] / 1000.0).astype(int))
    val_ids = set(np.floor(val[:, 0] / 1000.0).astype(int))
    assert train_ids.isdisjoint(val_ids)
    assert len(train) == 8 * 21
    assert len(val) == 2 * 21


def test_precomputed_windows_are_split_by_row(tmp_path):
    row_ids = np.arange(10, dtype=np.float64)[:, None]
    time = np.arange(80, dtype=np.float64)[None, :]
    windows = 1000.0 * row_ids + time
    path = tmp_path / "windows.npz"
    np.savez(path, data=windows)

    train, val = load_real_window_splits(
        real_data_cfg(path), window_length=80, context_length=64
    )

    train_ids = set(np.floor(train[:, 0] / 1000.0).astype(int))
    val_ids = set(np.floor(val[:, 0] / 1000.0).astype(int))
    assert train_ids.isdisjoint(val_ids)
    assert len(train) == 8
    assert len(val) == 2


def test_single_series_uses_non_overlapping_contiguous_segments(tmp_path):
    signal = np.arange(400, dtype=np.float64)
    path = tmp_path / "single.npz"
    np.savez(path, data=signal)

    train, val = load_real_window_splits(
        real_data_cfg(path), window_length=80, context_length=64
    )

    assert train[:, -1].max() < val[:, 0].min()

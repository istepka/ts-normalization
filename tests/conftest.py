"""Shared fixtures for the tsfm_pretraining test suite.

Builds tiny synthetic GiftEvalPretrain-shaped dataset directories (same
on-disk layout and Arrow schema as the real corpus: item_id/start/freq/target
columns, dataset_info.json, state.json) so tests run fast and don't depend on
the real 836GB corpus being mounted at a specific cluster path. The real
corpus is exercised directly by the Phase 1/2/5 smoke tests instead (see
notes/agentic_logs).
"""

from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pytest


def _build_dataset(
    root: Path, name: str, freq: str, n_series: int, length: int, seed: int
) -> None:
    targets = [
        (10.0 * (i + 1) + np.sin(np.arange(length, dtype=np.float32) / 5.0 + i)).astype(
            np.float32
        )
        for i in range(n_series)
    ]
    ds = datasets.Dataset.from_dict(
        {
            "item_id": [f"{name}_{i}" for i in range(n_series)],
            "start": [pd.Timestamp("2020-01-01")] * n_series,
            "freq": [freq] * n_series,
            "target": targets,
        },
        features=datasets.Features(
            {
                "item_id": datasets.Value("string"),
                "start": datasets.Value("timestamp[s]"),
                "freq": datasets.Value("string"),
                "target": datasets.Sequence(datasets.Value("float32")),
            }
        ),
    )
    ds.save_to_disk(str(root / name))


def _build_multivariate_dataset(
    root: Path, name: str, freq: str, n_series: int, length: int, n_channels: int
) -> None:
    targets = [
        [
            (10.0 * (i + 1) + np.arange(length, dtype=np.float32)).tolist()
            for _ in range(n_channels)
        ]
        for i in range(n_series)
    ]
    ds = datasets.Dataset.from_dict(
        {
            "item_id": [f"{name}_{i}" for i in range(n_series)],
            "start": [pd.Timestamp("2020-01-01")] * n_series,
            "freq": [freq] * n_series,
            "target": targets,
        },
        features=datasets.Features(
            {
                "item_id": datasets.Value("string"),
                "start": datasets.Value("timestamp[s]"),
                "freq": datasets.Value("string"),
                "target": datasets.Sequence(
                    datasets.Sequence(datasets.Value("float32")), length=n_channels
                ),
            }
        ),
    )
    ds.save_to_disk(str(root / name))


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> tuple[Path, dict]:
    """A 2-dataset synthetic corpus: synth_a (daily, 8 series x 200pts) and
    synth_b (5-minute, 6 series x 200pts), plus one multivariate dataset
    (synth_mv) that must be excluded from univariate iteration."""
    root = tmp_path / "gifteval_synth"
    root.mkdir()
    _build_dataset(root, "synth_a", "D", n_series=8, length=200, seed=0)
    _build_dataset(root, "synth_b", "5T", n_series=6, length=200, seed=1)
    _build_multivariate_dataset(
        root, "synth_mv", "H", n_series=4, length=100, n_channels=3
    )
    domain_map = {
        "synth_a": {"domain": "Nature", "confidence": "high"},
        "synth_b": {"domain": "Transport", "confidence": "high"},
        "synth_mv": {"domain": "Energy", "confidence": "high"},
    }
    return root, domain_map

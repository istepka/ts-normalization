"""Canonical shared window index over the univariate GiftEvalPretrain corpus.

Both the MOMENT and TimesFM stages consume the same base index so that, within a
paired natural-mixture or controlled-scale run, both models see identical windows
in identical order; only the model-specific masking/target construction and the
loss space differ. See notes/05-timesfm-pretraining-loss-space-plan.md.

Every window is `context_length + prediction_length` raw points sliced from one
series. MOMENT's masked-reconstruction task operates over the `context_length`
portion (its `seq_len`); TimesFM uses `context_length` as decoder input and
`prediction_length` as the forecast horizon. Both derive their own masking /
augmentation randomness deterministically from the window's `mask_seed` /
`aug_seed` (stable hashes of `(base_seed, dataset, series_id, window_start)`), so
replaying a seed reproduces identical batches.

The "valid-value mask" and "missing-value mask" required per window/series by the
plan are represented as reconstructable derived quantities (a stored
`valid_fraction` summary plus the ability to regenerate the exact boolean mask
on demand from the source Arrow row) rather than duplicated boolean arrays in the
index table, for the same storage reason documented in gifteval_corpus.py: the
positions are already fully determined by `(dataset, series_id, window_start)` and
duplicating them for tens of millions of windows would be pure overhead.

Series-level train/val split and the scale-swap A/B complement partition are both
derived the same way: a stable hash of `(base_seed, salt, dataset, series_id[,
window_start])` compared against a threshold. This makes both partitions exact,
reproducible, and free of any stored permutation state.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch

from . import gifteval_corpus as gc

WINDOW_INDEX_VERSION = "v1"


def stable_seed(*parts: object) -> int:
    """Deterministic 63-bit seed from arbitrary parts (stable across processes)."""
    joined = ":".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def unit_interval(*parts: object) -> float:
    """Deterministic value in [0, 1) from arbitrary parts, for threshold splits."""
    return stable_seed(*parts) / 2**63


class SeriesCache:
    """Caches memory-mapped HF datasets and item_id -> row lookups per dataset name.

    Avoids reopening Arrow files or rebuilding id maps on every window fetch.
    """

    def __init__(self, corpus_root: Path):
        self.corpus_root = Path(corpus_root)
        self._datasets: dict[str, datasets.Dataset] = {}
        self._id_to_row: dict[str, dict[str, int]] = {}

    def dataset(self, name: str) -> datasets.Dataset:
        if name not in self._datasets:
            ds = datasets.load_from_disk(str(self.corpus_root / name))
            self._datasets[name] = ds.with_format("numpy")
        return self._datasets[name]

    def row_index(self, name: str, series_id: str) -> int:
        if name not in self._id_to_row:
            ds = self.dataset(name)
            self._id_to_row[name] = {
                str(item_id): i for i, item_id in enumerate(ds["item_id"])
            }
        return self._id_to_row[name][series_id]

    def target(self, name: str, series_id: str) -> np.ndarray:
        row = self.row_index(name, series_id)
        return np.asarray(self.dataset(name)[row]["target"], dtype=np.float32)


@dataclass(frozen=True)
class WindowIndexConfig:
    context_length: int = 512
    prediction_length: int = 128
    stride: int = 512
    val_series_fraction: float = 0.1
    min_valid_fraction: float = 0.9
    base_seed: int = 0
    max_windows_per_series: int | None = None


class WindowIndex:
    """A built window index: a table plus the config it was built with."""

    def __init__(
        self, table: pd.DataFrame, config: WindowIndexConfig, corpus_root: Path
    ):
        self.table = table
        self.config = config
        self.corpus_root = Path(corpus_root)

    def __len__(self) -> int:
        return len(self.table)

    def split(self, name: str) -> pd.DataFrame:
        if name not in ("train", "val"):
            raise ValueError(f"unknown split {name!r}")
        return self.table[self.table["split"] == name]

    def window_values(self, row: pd.Series, cache: SeriesCache) -> np.ndarray:
        target = cache.target(row["dataset"], row["series_id"])
        start = int(row["window_start"])
        end = start + self.config.context_length + self.config.prediction_length
        return target[start:end]

    def valid_mask(self, row: pd.Series, cache: SeriesCache) -> np.ndarray:
        return ~np.isnan(self.window_values(row, cache))

    def mask_generator(self, row: pd.Series) -> torch.Generator:
        return torch.Generator().manual_seed(int(row["mask_seed"]))

    def augmentation_generator(self, row: pd.Series) -> torch.Generator:
        return torch.Generator().manual_seed(int(row["aug_seed"]))

    def scale_for(
        self, row: pd.Series, assignment: str, b_low: float, b_high: float
    ) -> float:
        """Scale factor for this window under assignment 'A' or 'B'.

        Assignment A: scale_group 0 -> b_low, scale_group 1 -> b_high.
        Assignment B swaps every window's scale while leaving scale_group (and
        therefore every other property of the window) untouched, so A and B are
        exact complements by construction.
        """
        if assignment not in ("A", "B"):
            raise ValueError(f"unknown scale assignment {assignment!r}")
        low_group = 0 if assignment == "A" else 1
        return b_low if int(row["scale_group"]) == low_group else b_high

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_parquet(path, index=False)

    @classmethod
    def load(
        cls, path: Path, config: WindowIndexConfig, corpus_root: Path
    ) -> "WindowIndex":
        table = pd.read_parquet(path)
        # window_values() slices raw series with self.config.context_length /
        # prediction_length, not whatever length the cached rows were
        # actually built with -- a stale cache built for a different config
        # would otherwise silently produce misaligned windows instead of
        # failing. context_length/prediction_length are the only WindowIndex-
        # affecting fields read back at query time; the rest (stride, seeds,
        # val fraction, ...) are already fully baked into the cached rows.
        cached_context = table["context_length"].unique()
        cached_prediction = table["prediction_length"].unique()
        if list(cached_context) != [config.context_length] or list(cached_prediction) != [
            config.prediction_length
        ]:
            raise ValueError(
                f"cached window index at {path} was built with "
                f"context_length={list(cached_context)}, "
                f"prediction_length={list(cached_prediction)}, but the requested config is "
                f"context_length={config.context_length}, prediction_length={config.prediction_length}. "
                "Rebuild the cache at a different path or fix the requested config -- "
                "loading a mismatched cache would silently misalign windows."
            )
        return cls(table, config, corpus_root)


def build_batch_schedule(
    index: "WindowIndex",
    split: str,
    dataset_weights: dict[str, float],
    steps: int,
    batch_size: int,
    schedule_seed: int,
) -> np.ndarray:
    """A fixed [steps, batch_size] array of positional row indices into
    `index.split(split).reset_index(drop=True)`, built once so paired
    conditions (e.g. moment_normalized vs moment_original, or scale
    assignment A vs B) train on identical windows in identical order --
    only the loss space or the applied scale can differ between them.

    Two-stage stratified draw per example: pick a dataset by
    `dataset_weights` (balances dataset identity), then a series uniformly
    within that dataset (balances series identity so series with many more
    windows don't dominate), then a window uniformly within that series.
    Domain/frequency balance is not forced directly -- realized exposure at
    those granularities is measured instead (see train.py's
    windows_processed counters), per the plan's "Log the cumulative number
    of windows actually processed" instruction.
    """
    table = index.split(split).reset_index(drop=True)
    if table.empty:
        raise ValueError(
            f"no windows in split={split!r} to build a batch schedule from"
        )
    datasets_present = sorted(table["dataset"].unique())
    missing = set(datasets_present) - set(dataset_weights)
    if missing:
        raise ValueError(f"dataset_weights missing entries for {sorted(missing)}")
    weights = np.array([dataset_weights[d] for d in datasets_present], dtype=np.float64)
    weights = weights / weights.sum()

    series_positions: dict[str, dict[str, np.ndarray]] = {
        dataset: {sid: sub.index.to_numpy() for sid, sub in group.groupby("series_id")}
        for dataset, group in table.groupby("dataset")
    }

    rng = np.random.default_rng(schedule_seed)
    n = steps * batch_size
    dataset_draws = rng.choice(len(datasets_present), size=n, p=weights)
    schedule = np.empty(n, dtype=np.int64)
    for d_idx, dataset in enumerate(datasets_present):
        draw_mask = dataset_draws == d_idx
        count = int(draw_mask.sum())
        if count == 0:
            continue
        series_map = series_positions[dataset]
        series_ids = list(series_map.keys())
        series_choice = rng.integers(0, len(series_ids), size=count)
        picks = np.empty(count, dtype=np.int64)
        for i, s_idx in enumerate(series_choice):
            positions = series_map[series_ids[s_idx]]
            picks[i] = positions[rng.integers(0, len(positions))]
        schedule[draw_mask] = picks
    return schedule.reshape(steps, batch_size)


def _series_split(
    base_seed: int, dataset: str, series_id: str, val_fraction: float
) -> str:
    return (
        "val"
        if unit_interval(base_seed, "split", dataset, series_id) < val_fraction
        else "train"
    )


def _series_scale_group(
    base_seed: int, dataset: str, frequency: str, series_id: str
) -> int:
    return (
        0
        if unit_interval(base_seed, "scale", dataset, frequency, series_id) < 0.5
        else 1
    )


def build_window_index(
    corpus_root: Path,
    dataset_names: list[str],
    domain_map: dict[str, dict],
    config: WindowIndexConfig,
) -> WindowIndex:
    """Scans the given univariate datasets and builds the canonical window index.

    Splits series into train/val before creating windows (no series appears in
    both), skips windows with more missing values than `min_valid_fraction`
    allows, and assigns each surviving series to a scale-swap complement group.
    """
    corpus_root = Path(corpus_root)
    window_length = config.context_length + config.prediction_length
    rows: list[dict] = []

    for name in dataset_names:
        info = gc.describe_dataset(
            corpus_root, name, domain_map, compute_checksum=False
        )
        if not info.is_univariate:
            continue
        ds = datasets.load_from_disk(str(corpus_root / name)).with_format("numpy")
        n_series = len(ds)
        # Batched slicing (matches gifteval_corpus.iter_series_records) instead
        # of one ds[i] access per series: on buildings_900k (1.8M series) the
        # per-row path did not finish scanning in 90s, this does the same work
        # in low single-digit seconds -- see notes/agentic_logs.
        for batch_start in range(0, n_series, gc.BATCH_SIZE):
            batch = ds[batch_start : batch_start + gc.BATCH_SIZE]
            item_ids, freqs, targets = batch["item_id"], batch["freq"], batch["target"]
            for i in range(len(item_ids)):
                series_id = str(item_ids[i])
                frequency = str(freqs[i])
                target = np.asarray(targets[i], dtype=np.float32)
                n = target.shape[0]
                if n < window_length:
                    continue

                split = _series_split(
                    config.base_seed, name, series_id, config.val_series_fraction
                )
                scale_group = _series_scale_group(
                    config.base_seed, name, frequency, series_id
                )

                starts = list(range(0, n - window_length + 1, config.stride))
                if config.max_windows_per_series is not None:
                    starts = starts[: config.max_windows_per_series]

                for window_start in starts:
                    window = target[window_start : window_start + window_length]
                    valid_fraction = float((~np.isnan(window)).mean())
                    if valid_fraction < config.min_valid_fraction:
                        continue
                    context = window[: config.context_length]
                    valid_context = context[~np.isnan(context)]
                    if valid_context.size == 0:
                        continue
                    context_mean = float(valid_context.mean())
                    context_std = float(valid_context.std())
                    if context_std <= 0.0 or not np.isfinite(context_std):
                        continue

                    rows.append(
                        {
                            "dataset": name,
                            "domain": info.domain,
                            "frequency": frequency,
                            "series_id": series_id,
                            "window_start": window_start,
                            "context_length": config.context_length,
                            "prediction_length": config.prediction_length,
                            "valid_fraction": valid_fraction,
                            "context_mean": context_mean,
                            "context_std": context_std,
                            "mask_seed": stable_seed(
                                config.base_seed, "mask", name, series_id, window_start
                            ),
                            "aug_seed": stable_seed(
                                config.base_seed, "aug", name, series_id, window_start
                            ),
                            "scale_group": scale_group,
                            "split": split,
                        }
                    )

    if not rows:
        raise ValueError(
            "window index is empty; check context/prediction length against the "
            "requested datasets' series lengths"
        )
    table = pd.DataFrame.from_records(rows)
    return WindowIndex(table, config, corpus_root)

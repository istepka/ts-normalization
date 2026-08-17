"""Enumerates the local GiftEvalPretrain corpus into per-series metadata records.

GiftEvalPretrain (https://huggingface.co/datasets/Salesforce/GiftEvalPretrain) is
stored on disk as one `datasets.Dataset.save_to_disk` directory per named dataset
(e.g. `PEMS03/`, `fred_md/`), each holding an Arrow file with columns `item_id`,
`start` (timestamp), `freq`, and `target`. A dataset is univariate if `target` is a
flat float32 sequence per series and multivariate if it is a fixed-length sequence
of float32 sequences (one inner sequence per channel); this is readable directly
from `dataset_info.json` without touching the Arrow data.

Domain is not present in the corpus at all (GiftEvalPretrain's card only states it
spans seven domains in aggregate); `gifteval_domains.yaml` supplies a manually
curated dataset -> domain assignment with a per-entry confidence flag, checked at
audit time to fail loudly if the local corpus contains a directory that mapping
does not cover.

"Missing-value mask" (as requested by the pretraining plan) is recorded per series
as a compact summary (`has_missing`, `missing_fraction`) rather than a duplicated
boolean array: the exact positions are always cheaply recoverable from the target
array itself via `~isnan(target)` given `(dataset, series_id)`, so storing the full
mask in the index would only duplicate data already on disk.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import datasets
import numpy as np
import yaml

PREPROCESSING_VERSION = "v1"
VALID_DOMAINS = frozenset(
    {"Econ/Fin", "Energy", "Healthcare", "Nature", "Sales", "Transport", "Web/CloudOps"}
)
DOMAIN_MAP_PATH = Path(__file__).parent / "gifteval_domains.yaml"
NON_DATASET_ENTRIES = frozenset({"README.md", ".cache", ".lts_corpus_index"})
BATCH_SIZE = 20_000


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    domain: str
    domain_confidence: str
    is_univariate: bool
    num_channels: int
    num_series: int
    checksum: str


@dataclass
class SeriesRecord:
    dataset: str
    domain: str
    frequency: str
    series_id: str
    start: str
    target_length: int
    has_missing: bool
    missing_fraction: float
    variance: float
    source_checksum: str
    preprocessing_version: str


def load_domain_map(path: Path = DOMAIN_MAP_PATH) -> dict[str, dict]:
    raw = yaml.safe_load(path.read_text())["datasets"]
    for name, entry in raw.items():
        if entry["domain"] not in VALID_DOMAINS:
            raise ValueError(f"{path}: unknown domain {entry['domain']!r} for {name}")
    return raw


def discover_dataset_dirs(
    corpus_root: Path, exclude: list[str] | None = None
) -> list[str]:
    """Dataset directory names under `corpus_root`, minus `exclude`.

    `exclude` holds datasets that must never be trained on because they are
    held out for evaluation (M1 and M3 by default, see conf/tsfm_base.yaml).
    An excluded name that is not present under `corpus_root` is an error: a
    typo there would silently reinstate the very leakage the list prevents.
    """
    corpus_root = Path(corpus_root)
    if not corpus_root.is_dir():
        raise FileNotFoundError(
            f"GiftEvalPretrain corpus root not found: {corpus_root}"
        )
    names = sorted(
        p.name
        for p in corpus_root.iterdir()
        if p.is_dir()
        and p.name not in NON_DATASET_ENTRIES
        and not p.name.startswith(".")
    )
    if not names:
        raise FileNotFoundError(f"No dataset directories found under {corpus_root}")
    if exclude:
        unknown = sorted(set(exclude) - set(names))
        if unknown:
            raise ValueError(
                f"corpus.exclude names not found under {corpus_root}: {unknown}"
            )
        names = [n for n in names if n not in set(exclude)]
    return names


def _arrow_files(dataset_dir: Path) -> list[Path]:
    files = sorted(dataset_dir.glob("data-*.arrow"))
    if not files:
        raise FileNotFoundError(f"No Arrow data files found in {dataset_dir}")
    return files


def checksum_dataset(dataset_dir: Path) -> str:
    """sha256 over the sorted Arrow file contents (source file checksum)."""
    digest = hashlib.sha256()
    for f in _arrow_files(dataset_dir):
        digest.update(f.read_bytes())
    return digest.hexdigest()


def read_dataset_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset_info.json in {dataset_dir}")
    return json.loads(info_path.read_text())


# The real GiftEvalPretrain corpus was written with an older `datasets`
# version that names a nested sequence feature "Sequence"; the currently
# installed `datasets` (5.x) renamed the same schema concept to "List" in its
# own dataset_info.json output. Both spellings mean the same thing here.
_SEQUENCE_TYPE_NAMES = frozenset({"Sequence", "List"})


def is_univariate(info: dict) -> tuple[bool, int]:
    """Returns (is_univariate, num_channels) from a dataset_info.json dict."""
    target_feature = info["features"]["target"]
    inner = target_feature["feature"]
    if inner["_type"] in _SEQUENCE_TYPE_NAMES:
        return False, int(target_feature["length"])
    return True, 1


def count_series(dataset_dir: Path) -> int:
    """Row count via a memory-mapped open of the Arrow file(s) (cheap; does not
    materialize series data). Independent of dataset_info.json's split
    metadata, which is only populated when a dataset was saved through a
    generator-based builder (true for the real corpus, not guaranteed for
    hand-built fixtures)."""
    return sum(
        datasets.Dataset.from_file(str(f)).num_rows for f in _arrow_files(dataset_dir)
    )


def describe_dataset(
    corpus_root: Path,
    name: str,
    domain_map: dict[str, dict],
    *,
    compute_checksum: bool = True,
) -> DatasetInfo:
    """`compute_checksum=False` skips hashing the Arrow file(s) -- full-file
    SHA256 is the correct provenance record for the audit script, but reading
    every byte of a large dataset (buildings_900k alone is 58GB) is a real,
    measured cost (observed >60s) that callers who only need the
    univariate/channel-count/domain metadata, like window_index.py, should
    not pay on every window index rebuild."""
    corpus_root = Path(corpus_root)
    dataset_dir = corpus_root / name
    if name not in domain_map:
        raise KeyError(
            f"{name!r} has no entry in {DOMAIN_MAP_PATH}; "
            "add one before it can be included in the corpus index"
        )
    info = read_dataset_info(dataset_dir)
    univariate, num_channels = is_univariate(info)
    return DatasetInfo(
        name=name,
        domain=domain_map[name]["domain"],
        domain_confidence=domain_map[name]["confidence"],
        is_univariate=univariate,
        num_channels=num_channels,
        num_series=count_series(dataset_dir),
        checksum=checksum_dataset(dataset_dir) if compute_checksum else "",
    )


def iter_series_records(corpus_root: Path, name: str, dataset_info: DatasetInfo):
    """Yields SeriesRecord for every series in a univariate dataset directory.

    Batches through the Arrow-backed HF dataset with numpy formatting so memory
    stays bounded regardless of dataset size (GiftEvalPretrain has datasets with
    millions of series).
    """
    if not dataset_info.is_univariate:
        return
    ds = datasets.load_from_disk(str(Path(corpus_root) / name))
    ds = ds.with_format("numpy")
    n = len(ds)
    for start in range(0, n, BATCH_SIZE):
        batch = ds[start : start + BATCH_SIZE]
        item_ids = batch["item_id"]
        starts = batch["start"]
        freqs = batch["freq"]
        targets = batch["target"]
        for i in range(len(item_ids)):
            target = np.asarray(targets[i], dtype=np.float64)
            missing = np.isnan(target)
            valid = target[~missing]
            yield SeriesRecord(
                dataset=name,
                domain=dataset_info.domain,
                frequency=str(freqs[i]),
                series_id=str(item_ids[i]),
                start=str(starts[i]),
                target_length=int(target.shape[0]),
                has_missing=bool(missing.any()),
                missing_fraction=float(missing.mean()) if target.shape[0] else 0.0,
                variance=float(valid.var()) if valid.size else float("nan"),
                source_checksum=dataset_info.checksum,
                preprocessing_version=PREPROCESSING_VERSION,
            )

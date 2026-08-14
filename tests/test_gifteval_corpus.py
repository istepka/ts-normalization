import datasets
import numpy as np
import pandas as pd

from src.data.gifteval import corpus as gc


def test_domain_map_covers_real_corpus_exactly():
    """gifteval_domains.yaml must exactly cover the real GiftEvalPretrain
    directories with no gaps -- gifteval_corpus.describe_dataset fails loudly
    on any directory missing an entry, so a stale mapping breaks the audit,
    not silently mis-labels a dataset."""
    domain_map = gc.load_domain_map()
    for name, entry in domain_map.items():
        assert entry["domain"] in gc.VALID_DOMAINS, name
        assert entry["confidence"] in ("high", "medium", "low"), name


def test_source_and_frequency_inventory_reproducible(tiny_corpus):
    root, domain_map = tiny_corpus
    names = gc.discover_dataset_dirs(root)
    assert names == sorted(names)

    infos_first = {n: gc.describe_dataset(root, n, domain_map) for n in names}
    infos_second = {n: gc.describe_dataset(root, n, domain_map) for n in names}
    assert infos_first == infos_second


def test_univariate_vs_multivariate_detection(tiny_corpus):
    root, domain_map = tiny_corpus
    a = gc.describe_dataset(root, "synth_a", domain_map)
    mv = gc.describe_dataset(root, "synth_mv", domain_map)
    assert a.is_univariate and a.num_channels == 1 and a.num_series == 8
    assert not mv.is_univariate and mv.num_channels == 3


def test_iter_series_records_skips_multivariate(tiny_corpus):
    root, domain_map = tiny_corpus
    mv_info = gc.describe_dataset(root, "synth_mv", domain_map)
    assert list(gc.iter_series_records(root, "synth_mv", mv_info)) == []


def test_series_record_fields(tiny_corpus):
    root, domain_map = tiny_corpus
    info = gc.describe_dataset(root, "synth_a", domain_map)
    records = list(gc.iter_series_records(root, "synth_a", info))
    assert len(records) == 8
    r = records[0]
    assert r.dataset == "synth_a"
    assert r.domain == "Nature"
    assert r.frequency == "D"
    assert r.target_length == 200
    assert r.has_missing is False
    assert r.missing_fraction == 0.0
    assert r.variance > 0
    assert r.preprocessing_version == gc.PREPROCESSING_VERSION


def test_missing_values_detected(tmp_path):
    rng = np.random.default_rng(0)
    target = np.arange(100, dtype=np.float32)
    target[rng.random(100) < 0.3] = np.nan
    ds = datasets.Dataset.from_dict(
        {
            "item_id": ["with_missing_0"],
            "start": [pd.Timestamp("2020-01-01")],
            "freq": ["D"],
            "target": [target],
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
    ds.save_to_disk(str(tmp_path / "with_missing"))

    domain_map = {"with_missing": {"domain": "Nature", "confidence": "high"}}
    info = gc.describe_dataset(tmp_path, "with_missing", domain_map)
    record = next(gc.iter_series_records(tmp_path, "with_missing", info))
    assert record.has_missing is True
    assert 0.0 < record.missing_fraction < 1.0

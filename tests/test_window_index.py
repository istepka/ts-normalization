import numpy as np
import pytest
import torch

from src.tsfm_pretraining import window_index as wi


def _build_index(tiny_corpus, **overrides):
    root, domain_map = tiny_corpus
    config = wi.WindowIndexConfig(
        context_length=32,
        prediction_length=8,
        stride=40,
        val_series_fraction=0.25,
        min_valid_fraction=0.9,
        base_seed=0,
        **overrides,
    )
    return wi.build_window_index(root, ["synth_a", "synth_b"], domain_map, config)


def test_window_index_skips_multivariate(tiny_corpus):
    index = _build_index(tiny_corpus)
    assert set(index.table["dataset"].unique()) == {"synth_a", "synth_b"}


def test_series_level_split_is_disjoint(tiny_corpus):
    index = _build_index(tiny_corpus)
    for dataset in index.table["dataset"].unique():
        sub = index.table[index.table["dataset"] == dataset]
        train_series = set(sub[sub["split"] == "train"]["series_id"])
        val_series = set(sub[sub["split"] == "val"]["series_id"])
        assert train_series.isdisjoint(val_series)
        assert len(val_series) > 0 and len(train_series) > 0


def test_scale_assignments_are_exact_complements(tiny_corpus):
    index = _build_index(tiny_corpus)
    for _, row in index.table.iterrows():
        a = index.scale_for(row, "A", b_low=1.0, b_high=10.0)
        b = index.scale_for(row, "B", b_low=1.0, b_high=10.0)
        assert {a, b} == {1.0, 10.0}
        assert a != b


def test_replaying_a_seed_reproduces_identical_masking_randomness(tiny_corpus):
    index = _build_index(tiny_corpus)
    row = index.table.iloc[0]
    gen1 = index.mask_generator(row)
    gen2 = index.mask_generator(row)
    assert torch.equal(torch.rand(10, generator=gen1), torch.rand(10, generator=gen2))


def test_window_index_config_determines_identical_index(tiny_corpus):
    """Two independent builds with the same base_seed produce byte-identical
    tables (same split, same scale_group, same seeds) -- required for paired
    conditions to consume identical base windows."""
    index_a = _build_index(tiny_corpus)
    index_b = _build_index(tiny_corpus)
    assert index_a.table.equals(index_b.table)


def test_load_rejects_cache_built_for_a_different_config(tiny_corpus, tmp_path):
    root, _ = tiny_corpus
    index = _build_index(tiny_corpus)
    cache_path = tmp_path / "index.parquet"
    index.save(cache_path)

    mismatched = wi.WindowIndexConfig(
        context_length=index.config.context_length + 8,  # differs from what was cached
        prediction_length=index.config.prediction_length,
        stride=index.config.stride,
        base_seed=index.config.base_seed,
    )
    with pytest.raises(ValueError, match="was built with"):
        wi.WindowIndex.load(cache_path, mismatched, root)

    # loading with the matching config still works
    reloaded = wi.WindowIndex.load(cache_path, index.config, root)
    assert reloaded.table.equals(index.table)


def test_window_values_and_valid_mask_match_source_series(tiny_corpus):
    root, _domain_map = tiny_corpus
    index = _build_index(tiny_corpus)
    cache = wi.SeriesCache(root)
    row = index.table.iloc[0]
    values = index.window_values(row, cache)
    mask = index.valid_mask(row, cache)
    assert values.shape == (
        index.config.context_length + index.config.prediction_length,
    )
    assert mask.shape == values.shape
    assert mask.all()  # synthetic fixture series have no missing values


def test_build_batch_schedule_is_deterministic_and_balances_datasets(tiny_corpus):
    index = _build_index(tiny_corpus)
    weights = {"synth_a": 1.0, "synth_b": 1.0}
    sched1 = wi.build_batch_schedule(
        index, "train", weights, steps=20, batch_size=8, schedule_seed=7
    )
    sched2 = wi.build_batch_schedule(
        index, "train", weights, steps=20, batch_size=8, schedule_seed=7
    )
    assert np.array_equal(sched1, sched2)

    train_table = index.split("train").reset_index(drop=True)
    datasets_drawn = train_table.iloc[sched1.ravel()]["dataset"]
    fractions = datasets_drawn.value_counts(normalize=True)
    # Equal weights over 160 draws should land within a generous tolerance of 50/50.
    assert abs(fractions["synth_a"] - 0.5) < 0.15


def test_build_batch_schedule_rejects_unweighted_dataset(tiny_corpus):
    index = _build_index(tiny_corpus)
    try:
        wi.build_batch_schedule(
            index, "train", {"synth_a": 1.0}, steps=1, batch_size=1, schedule_seed=0
        )
    except ValueError as e:
        assert "synth_b" in str(e)
    else:
        raise AssertionError("expected ValueError for missing dataset weight")

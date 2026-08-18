import numpy as np
import pytest
import torch

from src.data.gifteval import window_index as wi


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


def test_scale_assignments_are_balanced_at_dataset_level(tiny_corpus):
    index = _build_index(tiny_corpus)
    scales = {}
    for dataset, rows in index.table.groupby("dataset"):
        assigned = {
            index.scale_for(row, "A", b_low=1.0, b_high=10.0)
            for _, row in rows.iterrows()
        }
        assert len(assigned) == 1
        scales[dataset] = assigned.pop()

    counts = np.unique(list(scales.values()), return_counts=True)[1]
    assert counts.max() - counts.min() <= 1


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
    with pytest.raises(ValueError, match="maximum geometry of"):
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


def _mixed_index(mixed_length_corpus, **overrides):
    root, domain_map = mixed_length_corpus
    config = wi.WindowIndexConfig(
        context_length=32,
        prediction_length=8,
        stride=40,
        val_series_fraction=0.25,
        min_valid_fraction=0.9,
        base_seed=0,
        **overrides,
    )
    return wi.build_window_index(root, ["mixed"], domain_map, config)


def test_geometry_rule_reproduces_the_fixed_geometry_for_long_series():
    """The whole point of the ratio rule: series that already produced
    windows must produce exactly the same ones, or the variable-geometry
    index is not a superset of the fixed one."""
    config = wi.WindowIndexConfig(
        context_length=512,
        prediction_length=128,
        min_context_length=64,
        min_prediction_length=8,
    )
    assert wi.series_geometry(640, config) == (512, 128)
    assert wi.series_geometry(5000, config) == (512, 128)
    assert wi.series_geometry(639, config) == (512, 127)


def test_geometry_rule_shrinks_short_series_and_rejects_the_too_short():
    config = wi.WindowIndexConfig(
        context_length=512,
        prediction_length=128,
        min_context_length=64,
        min_prediction_length=8,
    )
    assert wi.series_geometry(200, config) == (160, 40)
    # 72 is the shortest admissible series: the ratio would leave a 58-point
    # context, so the context pins to its minimum and the horizon takes 8.
    assert wi.series_geometry(72, config) == (64, 8)
    assert wi.series_geometry(71, config) is None


def test_minimum_geometry_must_be_set_as_a_pair():
    with pytest.raises(ValueError, match="together or not at all"):
        wi.WindowIndexConfig(min_context_length=64)


def test_variable_index_is_a_strict_superset_of_the_fixed_one(mixed_length_corpus):
    fixed = _mixed_index(mixed_length_corpus)
    variable = _mixed_index(
        mixed_length_corpus, min_context_length=8, min_prediction_length=2
    )
    key = ["dataset", "series_id", "window_start", "context_length"]
    fixed_keys = set(map(tuple, fixed.table[key].to_numpy().tolist()))
    variable_keys = set(map(tuple, variable.table[key].to_numpy().tolist()))
    assert fixed_keys <= variable_keys
    assert len(variable.table) > len(fixed.table)


def test_short_windows_are_padded_against_the_context_boundary(mixed_length_corpus):
    """A short window must land so the adapters' two fixed slice offsets
    still separate its context from its horizon, with padding read as
    missing on either side."""
    root, _ = mixed_length_corpus
    index = _mixed_index(
        mixed_length_corpus, min_context_length=8, min_prediction_length=2
    )
    cache = wi.SeriesCache(root)
    max_context = index.config.context_length
    short = index.table[index.table["context_length"] < max_context]
    assert len(short) > 0
    row = short.iloc[0]
    values = index.window_values(row, cache)
    mask = index.valid_mask(row, cache)
    assert values.shape == (max_context + index.config.prediction_length,)

    context_length = int(row["context_length"])
    prediction_length = int(row["prediction_length"])
    assert mask[max_context - context_length : max_context].all()
    assert not mask[: max_context - context_length].any()
    assert mask[max_context : max_context + prediction_length].all()
    assert not mask[max_context + prediction_length :].any()

    raw = cache.target(row["dataset"], row["series_id"])
    start = int(row["window_start"])
    expected = raw[start : start + context_length + prediction_length]
    assert np.array_equal(values[mask], expected)

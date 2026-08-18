"""The train-region cut applied to the held-out evaluation suites.

The one thing worth pinning is that this cut and the supervised one are the
same cut. They live in two files by explicit decision (see
notes/07-short-series-retrain-plan.md), and if they ever disagree the
comparison between the pretrained and supervised tables stops meaning
anything.
"""

import numpy as np
import pytest

from src.data.gifteval import train_regions as tr


def supervised_validation_start(canonical_length, official_horizon, validation_size):
    """`split_series` from src/supervised/data.py, transcribed.

    Kept as a literal transcription rather than an import: the supervised
    module lives on another branch, and a paraphrase would pin nothing.
    """
    test_size = 2 * official_horizon - 1
    test_start = canonical_length - test_size
    return test_start - validation_size


@pytest.mark.parametrize(
    "length,horizon,validation",
    [(120, 18, 18), (71, 18, 24), (9919, 14, 16), (40, 6, 8)],
)
def test_cut_matches_the_supervised_split(length, horizon, validation):
    assert tr.train_end_index(length, horizon, validation) == (
        supervised_validation_start(length, horizon, validation)
    )


def test_a_region_stops_before_every_scored_point():
    """The evaluated horizon is the final H points. The cut has to clear it
    by the whole reserved test and validation region, not merely miss it."""
    length, horizon, validation = 200, 18, 24
    end = tr.train_end_index(length, horizon, validation)
    first_scored = length - horizon
    assert end < first_scored
    assert first_scored - end == horizon - 1 + validation


def test_write_regions_groups_by_dataset(tmp_path):
    import datasets

    regions = [
        tr.TrainRegion(
            dataset="trainsplit_a",
            item_id=f"i{i}",
            freq="M",
            values=np.arange(10 + i, dtype=np.float32),
            canonical_length=100,
            official_horizon=18,
            validation_size=18,
            evaluated=True,
        )
        for i in range(3)
    ] + [
        tr.TrainRegion(
            dataset="trainsplit_b",
            item_id="j0",
            freq="Q",
            values=np.arange(7, dtype=np.float32),
            canonical_length=60,
            official_horizon=8,
            validation_size=8,
            evaluated=True,
        )
    ]
    counts = tr.write_regions(regions, tmp_path)
    assert counts == {"trainsplit_a": 3, "trainsplit_b": 1}

    written = datasets.load_from_disk(str(tmp_path / "trainsplit_a"))
    assert written.column_names == ["item_id", "freq", "target"]
    assert [len(t) for t in written["target"]] == [10, 11, 12]

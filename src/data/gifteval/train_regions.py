"""Train-region copies of the held-out evaluation suites, as corpus datasets.

M1, M3, M4, Tourism, and Favorita are scored held-out, so their canonical
series must never enter pretraining. Their *train regions* can, and once the
window index admits short series they are the only part of the corpus that
looks anything like what those suites are scored on.

The split convention is the supervised one, from `src/supervised/data.py` on
branch `feat/m-series-supervised`, so the pretrained and supervised tables
are cut from the same series the same way:

- the final `2H - 1` points are the rolling test region,
- the `validation_size` points before that are validation,
- everything earlier is the train region, and that is what is emitted here.

`H` is the series' own official competition horizon. `validation_size` is the
pooled model horizon, the largest official horizon among every M-series and
Tourism series of the same frequency, because that is what a pooled
supervised run reserves. Reserving the largest is what makes this safe under
either pooling, since over-reserving only costs training points.

The emitted datasets carry no `start` column. The window index never reads
one, the Monash `.tsf` reader does not parse it, and the M4 CSVs ship no
dates at all, so writing a placeholder timestamp would be inventing calendar
alignment that does not exist. `freq` is carried through, because the window
index does read that.
"""

from dataclasses import dataclass
from pathlib import Path

import datasets
import numpy as np
import pandas as pd

from src.eval import suites as eval_suites

FAVORITA_DATASET = "favorita_sales"
# Emitted names are prefixed so they can never collide with the canonical
# corpus directories the exclusion list names, which hold the full series.
PREFIX = "trainsplit_"


@dataclass(frozen=True)
class TrainRegion:
    """One series cut down to its train region."""

    dataset: str
    item_id: str
    freq: str
    values: np.ndarray
    canonical_length: int
    official_horizon: int
    validation_size: int
    # False only for Favorita series that stop before the evaluation cutoff.
    # Nothing in them is ever scored, so they are emitted whole and the audit
    # has no cut to check.
    evaluated: bool


def train_end_index(
    canonical_length: int, official_horizon: int, validation_size: int
) -> int:
    """Where the train region stops, exclusive."""
    return canonical_length - (2 * official_horizon - 1) - validation_size


def monash_and_m4_regions(monash_root: Path, m4_root: Path) -> list[TrainRegion]:
    """Train regions for M1, M3, Tourism, and M4.

    All four come from the eval loaders rather than from the corpus copies of
    the same competitions, so the series cut here are provably the series
    scored later: the loaders assert the published counts, and the corpus
    copies carry no such guarantee.
    """
    items = []
    for suite in ("m1", "m3", "tourism"):
        items.extend(eval_suites.load_monash(monash_root, suite))
    items.extend(eval_suites.load_m4(m4_root))

    validation_size = {}
    for item in items:
        horizon = len(item.actual)
        validation_size[item.freq] = max(validation_size.get(item.freq, 0), horizon)

    out = []
    for item in items:
        values = np.concatenate((item.history, item.actual))
        horizon = len(item.actual)
        end = train_end_index(len(values), horizon, validation_size[item.freq])
        if end <= 0:
            continue
        out.append(
            TrainRegion(
                dataset=f"{PREFIX}{_dataset_name(item)}",
                item_id=item.item_id,
                freq=item.freq,
                values=values[:end].astype(np.float32),
                canonical_length=len(values),
                official_horizon=horizon,
                validation_size=validation_size[item.freq],
                evaluated=True,
            )
        )
    return out


def favorita_regions(corpus_root: Path) -> list[TrainRegion]:
    """Train regions for Favorita.

    Read straight from the corpus copy, which is the same data the eval
    harness scores. Only series reaching `FAVORITA_END` are evaluated, and
    only those are cut; the rest are emitted whole, since nothing in them is
    ever scored.
    """
    dataset = datasets.load_from_disk(str(Path(corpus_root) / FAVORITA_DATASET))
    horizon = eval_suites.FAVORITA_HORIZON
    out = []
    n_eligible = 0
    for batch in dataset.iter(batch_size=5000):
        for item_id, start, target in zip(
            batch["item_id"], batch["start"], batch["target"]
        ):
            values = np.asarray(target, dtype=np.float32)
            end_date = pd.Timestamp(start) + pd.Timedelta(days=len(values) - 1)
            evaluated = end_date == eval_suites.FAVORITA_END
            if evaluated:
                n_eligible += 1
                end = train_end_index(len(values), horizon, horizon)
            else:
                end = len(values)
            if end <= 0:
                continue
            out.append(
                TrainRegion(
                    dataset=f"{PREFIX}{FAVORITA_DATASET}",
                    item_id=str(item_id),
                    freq="D",
                    values=values[:end],
                    canonical_length=len(values),
                    official_horizon=horizon,
                    validation_size=horizon,
                    evaluated=evaluated,
                )
            )
    expected = eval_suites.EXPECTED_SERIES["favorita"]
    if n_eligible != expected:
        raise ValueError(
            f"Favorita: {n_eligible} series reach {eval_suites.FAVORITA_END}, "
            f"expected {expected}; the corpus copy has drifted from the one "
            "the eval harness scores"
        )
    return out


def _dataset_name(item: eval_suites.EvalSeries) -> str:
    """The corpus directory an eval series' train region belongs in.

    Monash subsets are already directory names (`m1_monthly`). M4 subsets are
    the competition's capitalized frequency names, so they are lowered and
    qualified.
    """
    if item.suite == "m4":
        return f"m4_{item.subset.lower()}"
    return item.subset


def write_regions(regions: list[TrainRegion], out_root: Path) -> dict[str, int]:
    """Writes one Arrow dataset per emitted dataset name. Returns the counts."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[TrainRegion]] = {}
    for region in regions:
        by_dataset.setdefault(region.dataset, []).append(region)

    counts = {}
    features = datasets.Features(
        {
            "item_id": datasets.Value("string"),
            "freq": datasets.Value("string"),
            "target": datasets.Sequence(datasets.Value("float32")),
        }
    )
    for name, group in sorted(by_dataset.items()):
        ds = datasets.Dataset.from_dict(
            {
                "item_id": [r.item_id for r in group],
                "freq": [r.freq for r in group],
                "target": [r.values.tolist() for r in group],
            },
            features=features,
        )
        ds.save_to_disk(str(out_root / name))
        counts[name] = len(group)
    return counts

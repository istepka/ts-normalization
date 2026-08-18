# Variable-geometry windows and the held-out train regions, as built

Status: implemented 2026-08-18. Executes
[07-short-series-retrain-plan.md](07-short-series-retrain-plan.md), whose
motivation is in [06-variable-context-windows.md](06-variable-context-windows.md).

This note records the decisions the plan deliberately left open, and what
each one turned out to cost.

## The geometry rule

A series of `n` points gets

```
prediction = clip(n // 5, 8, 128)
context    = min(512, n - prediction)
if context < 64:  context, prediction = 64, n - 64
```

and contributes nothing if that leaves fewer than 64 context or 8 horizon
points, which is every series under 72 points.

The `// 5` keeps the maximum geometry's 4:1 context-to-horizon ratio, and
that is the whole reason for choosing it over a fixed short horizon. At
`n = 640` it returns exactly `(512, 128)`, so **every window the old index
held, the new one holds too, unchanged**. The variable index is a strict
superset of the fixed one rather than a different index, which is what makes
the old and new runs comparable at all. `test_geometry_rule_reproduces_the_
fixed_geometry_for_long_series` pins this.

The plan asked whether the horizon should instead be fixed at 8 for short
windows. It should not: a 100-point series would then train on a 92-point
context, which is nothing like the 4:1 shape every long window has, and the
model would see two unrelated tasks rather than one task at two scales.

## Ragged batching: pad to the fixed maximum, not to the batch maximum

Both options in the plan were pad-and-mask against a moving target. The
choice made here is neither: every window is padded out to the full 512 plus
128 shape, with the context right-aligned against the boundary at 512 and
the horizon left-aligned after it.

This is worth the wasted compute for three reasons.

**No adapter changes at all.** Chronos-2, Moirai-2.0, TimesFM, and MOMENT
already slice at two fixed offsets and already carry a validity mask for
missing values. Padding reads as missing. Not one line of the four adapters
moved.

**Per-example loss stays independent of the batch.** Padding to the batch
maximum would make a window's loss depend on which other windows were drawn
alongside it, and the paired conditions would then diverge for a reason that
has nothing to do with the loss space the paper is about.
`test_mixed_geometry_batch_scores_each_window_independently` runs a
mixed-length batch through Chronos-2 and asserts the per-example losses equal
what each window scores alone.

**It matches evaluation exactly.** `src/eval/predict.py` already left-pads a
short history into the model's context width. Training now does the same
thing to the same series, which is precisely the train/test mismatch note 06
identified.

The cost is real and unhedged: a 72-point window occupies a 640-wide tensor,
so the short end of the corpus is roughly 90% padding. Length bucketing would
recover that compute and can be added later without changing the index.

## Sampling

No new rule was needed. `build_batch_schedule` already draws dataset, then
series within dataset, then window within series, so a dataset's exposure is
set by its weight and not by how many windows its series happen to yield.
Short series produce about one window each and long ones many, and that
ratio never reaches the batch.

What was missing is the record. `window_index_meta.json` now carries a
`geometry` block with the window count and share per realized
`(context, prediction)` pair, because the length mix is part of what a
checkpoint means once the geometry stops being fixed.

## The train regions

`src/data/gifteval/train_regions.py` emits one Arrow dataset per held-out
suite subset, holding each series cut to `len - (2H - 1) - validation_size`.
`H` is the series' own official horizon; `validation_size` is the largest
official horizon among every M-series and Tourism series of the same
frequency, which is what a pooled supervised run reserves. Over-reserving
only costs training points, so taking the largest is what keeps this safe
whichever way the supervised runs pool.

M1, M3, Tourism, and M4 are cut from the **eval loaders**, not from the
corpus copies of the same competitions. The loaders assert the published
series counts, so the series cut here are provably the series scored later.
Favorita is read from its corpus directory, which is the same data the
harness scores.

The canonical copies stay in `corpus.exclude` and the emitted directories
carry a `trainsplit_` prefix, so the exclusion list did not have to move.
That sidesteps the plan's item 8 entirely: there is no window in which a
rebuilt index and a stale exclusion list could disagree.

Emitted datasets carry no `start` column. The window index never reads one,
the Monash `.tsf` reader does not parse it, and the M4 CSVs ship no dates at
all, so a placeholder timestamp would be inventing calendar alignment that
does not exist.

### What the audit found

The build audits every emitted series against its canonical length before
writing anything, and it caught a real case on the first run: Favorita series
that stop before the 2017-08-15 cutoff are absent from the suite the harness
loads, so they have no scored point to stop before and are emitted whole.
That is now an explicit `evaluated` flag rather than an implicit exception.

216,724 series emitted, 188,091 of them evaluated, minimum 15 points reserved
and median 47.

### What the train regions actually yield

Measured at 512+128 down to 64+8, stride 512:

| dataset | series | usable | windows |
|---|---|---|---|
| trainsplit_m4_monthly | 48,000 | 32,919 | 32,959 |
| trainsplit_m4_daily | 4,227 | 4,217 | 17,153 |
| trainsplit_m4_quarterly | 24,000 | 12,320 | 12,320 |
| trainsplit_m3_monthly | 1,428 | 799 | 799 |
| trainsplit_m4_weekly | 359 | 294 | 512 |
| trainsplit_m4_hourly | 414 | 414 | 414 |
| trainsplit_tourism_monthly | 366 | 365 | 365 |
| trainsplit_tourism_quarterly | 427 | 302 | 302 |
| trainsplit_m4_yearly | 22,765 | 165 | 165 |
| trainsplit_m1_monthly | 549 | 100 | 100 |
| trainsplit_m3_other | 174 | 22 | 22 |
| trainsplit_m1_quarterly | 177 | 4 | 4 |
| trainsplit_m1_yearly, m3_yearly, m3_quarterly, tourism_yearly | 1,998 | 0 | 0 |
| **total (excluding Favorita)** | **104,884** | **51,921** | **65,115** |

**The yearly competitions still contribute nothing, and cannot.** A yearly
series is a few dozen points, and reserving `3H - 1` of them leaves under the
72 a window needs. This is not a threshold that can be tuned down without
abandoning the idea that a context should carry more information than the
horizon it predicts.

65,115 windows is small against 41 million, and the plan expected the
geometry change to be the larger of the two effects. It is not, for the
reason the next section records. What carries this rerun is the sampling
weight these 13 datasets get, not the windows they contribute.

## The rebuilt index, and a correction to note 06

Built 2026-08-18 as job 32794, 41 minutes over 157 dataset directories.

| | fixed 512+128 | variable, down to 64+8 |
|---|---|---|
| windows | 40,952,582 | 41,260,459 |
| datasets with windows | 53 | 73 |
| sub-512 windows | 0 | 231,263 (0.56%) |

**Note 06 was wrong about why the corpus was invisible.** It said 99 dataset
directories contributed nothing "because no series in them is long enough."
Of the 84 still contributing nothing at 64 plus 8, **78 are multivariate**,
which `build_window_index` skips by design and which no geometry change was
ever going to admit. Only six are univariate and genuinely too short:
`cif_2016_6`, `rideshare_with_missing`, and the four yearly train-split
suites. The corpus was never mostly short, it was mostly gridded (`era5_*`
and `cmip6_*` alone are 75 of the 78).

So the geometry change bought 20 datasets, not 99, and 0.75% more windows.
Five are ordinary corpus datasets that were just under the old threshold
(`cif_2016_12`, `covid_mobility`, `kaggle_web_traffic_weekly`, `nn5_weekly`,
`traffic_weekly`, `uber_tlc_daily`, `vehicle_trips_with_missing`). Thirteen
are the train regions.

### Window share is not exposure share

The train regions are 0.31% of the index and **17.8% of the sampling
weight**, because `build_batch_schedule` draws a dataset first and
`dataset_weights` is uniform. Thirteen of 73 datasets means roughly one
training example in six comes from a held-out suite's train region.

That is a deliberate consequence of the existing sampler, not an accident,
and it is almost certainly the largest single effect in this rerun: it
matters far more than the 0.75% more windows. It may well be too much. If
the short-series suites improve sharply while GIFT-Eval regresses, this
weight is the first thing to look at, and `dataset_weights` is where to
change it without rebuilding anything.

## What these runs are not

**They are not zero-shot on M1, M3, M4, Tourism, or Favorita.** Those suites
are in-domain now. The clean-corpus runs (jobs 32459 and 32462) stay as the
zero-shot table `notes/PLAN.md` asks for, and the two sets of checkpoints
have to be labelled wherever they are reported together. Neither the eval
reports nor the artifact carries a field distinguishing them yet.

Controlled-scale results cannot be pooled across this index rebuild, since
`_dataset_scale_group` splits the stable-hash-ordered dataset list at
`len // 2` and that split moves when the dataset set changes. Only
natural-scale runs are in scope here.

# Natural-scale re-run for Chronos-2 and Moirai-2.0 (2026-08-17)

## Why re-run

The 2026-08-12 main runs (`gifteval_chronos2_b512_29436`,
`gifteval_moirai2_b512_29437`) predate four changes that all touch the loss
being measured:

- `1b81955` fixed Moirai evaluation dropout.
- `98abb48` moved Moirai 2.0 and TimesFM onto their papers' objectives.
- `51eaf55` / `261cce1` / `9af9794` gave the adapters a shared
  scale-invariant normalization module.
- `7bfd67b` held M1, M3, Tourism, and Favorita out of the corpus, which
  changed the window index.

## Two blockers found before submitting

**The launcher's default index no longer trains.** `resolve_window_index`
raises when a cached index contains a held-out dataset, and Favorita
contributes 78,034 windows to the old
`context512_pred128.parquet`. Every launcher defaulted to that file, so the
default path was a hard failure. All three now default to
`context512_pred128_heldout.parquet`, the index built as job 31052. The
build sbatch's default output name moved too, since its default `--exclude`
is non-empty and an index built there is always a held-out one. The old
index stays on disk so the completed runs remain reproducible.

**There was no way to ask for natural scale only.** Every model's `RUNS`
list is two `natural_mixture` entries followed by four `controlled_scale`
ones, so `NATURAL_ONLY=1` truncates to the first two. Array bounds shrink to
match in `submit_pretraining.sh` via `RUNS_PER_SEED`.

## Scope

Natural scale only, four seeds, both loss conditions. Eight tasks per model.

| task | run dir |
|---|---|
| 0, 1 | `seed0_{model}_{normalized,original}_natural` |
| 2, 3 | `seed1_{model}_{normalized,original}_natural` |
| 4, 5 | `seed2_{model}_{normalized,original}_natural` |
| 6, 7 | `seed3_{model}_{normalized,original}_natural` |

## Steps and wall clock

60,000 steps at batch size 512, reached directly rather than as a 30k run
plus the separate continuation the 2026-08-13 campaign used. Measured 30k
elapsed at b512 was 2:30 for Chronos-2 and 1:48 for Moirai-2.0, so 60k
projects to roughly 5:00 and 3:36, both inside the launcher's existing
12-hour limit. No `--time` change was needed.

## Chained evaluation

`submit_pretraining.sh` now chains two jobs onto every pretraining array,
which `EVAL=0` skips and MOMENT is excluded from for want of a forecast head:

1. `scripts/eval_pretraining.sbatch`, a GPU array with one task per run,
   `afterok` the pretraining array. It discovers run directories by globbing
   the run root rather than reconstructing the `RUNS` table, so it stays
   correct under `NATURAL_ONLY` and any seed list.
2. `scripts/collect_eval.sbatch`, `afterok` that array, merging the per-run
   tables into the reports below.

**The eval job reads each run's own `resolved_config.yaml`, never
`conf/eval.yaml`'s model skeleton.** This is not a nicety. The 2026-08-12
Chronos-2 runs were trained at `d_ff: 2048, dropout_rate: 0.1` while
`conf/eval.yaml` declares `1024` and `0.0`, so rebuilding those checkpoints
from the eval defaults fails on a shape mismatch. A checkpoint only loads
into the architecture that produced it, so the run's config is the only
source of truth for the skeleton.

## Reports

`src/scripts/collect_tsfm_eval.py` writes both table grains, each as CSV and
markdown, always as SIT against RevIN:

| grain | file | for |
|---|---|---|
| benchmark, everything else collapsed | `eval_report_main.md` | main paper |
| benchmark by frequency | `eval_report_by_frequency.md` | appendix |

`NormalizationModule.module` maps `<model>_normalized` to `SIT` and
`<model>_original` to `RevIN`, so the run conditions are those two labels
under different names. TimesFM's original-space condition is
`timesfm_native_original`, which is why the mapping tests the suffix rather
than the whole string.

Subsets are weighted by series count within a benchmark row, since a flat
mean over GIFT-Eval's 55 configs is a different claim than a mean over its
instances. Seeds are averaged rather than pooled, each being an independent
training run, and the standard deviation across them is reported alongside.
Every metric is lower-is-better, so a ratio below 1 favors SIT throughout.

## Launch

```bash
scripts/main_paper_tsfm_experiment_submit.sh
```

That wrapper is the whole experiment definition, so the seed list and step
count live in one place rather than in a command someone has to retype
correctly. It expands to one `submit_pretraining.sh` call per model:

```bash
MODEL=<model> NATURAL_ONLY=1 SEEDS_CSV=0,1,2,3 STEPS=60000 \
  scripts/submit_pretraining.sh
```

Chronos-2 and Moirai-2.0 are not auto-aggregated, so no loss-space
aggregation job is submitted alongside them, only the eval chain.

This is also the first time the eval harness meets a real checkpoint, which
closes the verification gap recorded in
`notes/agentic_logs/2026-08-16-eval-harness.md`. The first thing to check in
the output is `fixed` mode against each run's own training-time eval loss.

## Comparability caveat

Rebuilding the index moved `nn5_daily_with_missing` between scale groups,
because `_dataset_scale_group` splits the dataset list at `len // 2` and the
corpus went from 55 datasets to 53. That only affects the controlled-scale
conditions, which this re-run does not include.

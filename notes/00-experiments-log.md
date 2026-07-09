# 00 — Experiments log (Slurm job ledger)

Brief record of which Slurm job was which, so results trace back to a run. Outputs live in
`outputs/<jobid>_loss_space_toy[_paper]/` (jobs before the job-ID-prefix change wrote to the
unprefixed `outputs/loss_space_toy/`). Spread = per-category nMSE max/min for
`[normalized, original, equalvar, gradmatch]`. All dates 2026-06-11.

## Canonical runs (cite these)

| Job | Config | Metric / code stage | Spreads (norm/orig/eqvar/gradmatch) | Status |
|-----|--------|---------------------|--------------------------------------|--------|
| **12638** | 30k, SGD | v5 — dense early eval (every 5 to 200), linear aggregation for linear plots | 4.09 / **292** / 6.01 / 71 | ✅ canonical SGD 30k |
| **12646** | 30k, Adam | v5 — same as 12638 | 2.77 / **3.46** / 1.45 / 2.97 | ✅ canonical Adam 30k |
| **12568** | 100k, SGD | v2 — held-out val (no multi-zoom/metrics npz) | 2.38 / **264** / 2.72 / 29 | ✅ canonical SGD 100k |

Headline: SGD original-space spread ~260–300× (fan-out by σ²). Adam does **not** remove the
bias — the σ²-ordered rate disparity is still there in the first ~150 steps (var1 ~125 vs
var100 ~5 steps to floor); its small final-step spread (~3.5×) is a saturation/overfitting
artifact, not a fix (read the linear early-window zoom). At 100k the SGD gap stays open.

## Superseded / earlier runs

| Job | Config | Why superseded | Spreads | Status |
|-----|--------|----------------|---------|--------|
| 12507 | 30k, SGD | first clean reference; pre job-ID-prefix (→ `outputs/loss_space_toy/`) | ~3.65 / 261 / 5.58 / 79 | ✅ done (ref) |
| 12521 | 30k, SGD | v1 minibatch metric; reproduced 12507 with prefixed output dir | 3.65 / 261 / 5.58 / 79 | ✅ done, superseded by 12563→12582 |
| 12547 | 30k, Adam | v1 minibatch metric | 1.30 / 3.17 / 4.16 / 2.07 | ✅ done, superseded by 12564→12583 |
| 12544 | 100k, SGD | old code (minibatch metric, buggy linear bands, no val set) | 2.71 / 234 / 3.06 / 20.6 | ✅ done, superseded by 12568 |
| 12563 | 30k, SGD | v2 (held-out val); only plotting changed vs v3 | 4.05 / 296 / 5.98 / 80 | ✅ done, superseded by 12582 |
| 12564 | 30k, Adam | v2 (held-out val); only plotting changed vs v3 | 1.72 / 2.23 / 1.86 / 1.79 | ✅ done, superseded by 12583 |
| 12582 | 30k, SGD | v3 (uniform 100-step eval); reproduced exactly by v5 dense-eval | 4.05 / 296 / 5.98 / 80 | ✅ done, superseded by 12638 |
| 12583 | 30k, Adam | v3 (uniform 100-step eval); same conclusion as v5 12646 | 1.72 / 2.23 / 1.86 / 1.79 | ✅ done, superseded by 12646 |

## Aborted / non-runs

| Job | Config | What happened |
|-----|--------|---------------|
| 12502–12506 | CPU/30k, SGD | early exploratory/debug (lr sweep, NaN diagnosis at 1:100:10⁴ spread, dup-submission); 12504 was the accidental duplicate that prompted the dup-guard |
| 12523 | 100k, SGD | **hung at startup** (froze after wandb login, 0 progress in 60 min — concurrent-launch fluke), cancelled |
| 12565, 12566 | 100k, SGD | **dup-guard exits** of 12544 (same job-name, higher JobID → exited without running) |

## Code-stage legend

- **v1 (minibatch metric)**: per-step nMSE measured on the 8-window training batch → noisy.
- **v2**: held-out validation metric (`val_windows_per_category` unseen-phase windows, fixed
  `VAL_SEED`); log-space mean±band fix; linear companions; bold forecast-evolution grid.
- **v3**: adds per-window linear zooms (`plot.linear_xlim: [200, 500, 2000]`) and
  `metrics_*.npz` persistence → figures replottable without retraining
  (`scripts/replot_metrics.py`).
- **v5**: non-uniform `eval_schedule` (every 5 steps to 200, every 50 to 1000, every 1000
  after) → smooth early curves; linear plots use arithmetic mean±std (honest absolute scale)
  while log plots keep geometric mean±log-std; `grad_magnitude_linear` companion added.

## W&B

Project `ts-normalization/loss_space_comparison`; groups `{experiment}/{label}`. Experiment
tags: `var_1_10_100_30k_v5` (12638), `var_1_10_100_30k_adam_v5` (12646),
`var_1_10_100_100k_v2` (12568); earlier batches used the `_v3` / `_v2` / un-suffixed variants.
v5 100k (12643) in flight — promote when it lands.

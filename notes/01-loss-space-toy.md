# 01 — Loss-space comparison toy

Demonstrates the §3 claim: computing the training loss in the **original** space scales
the gradient by `b^p = σ^p` relative to the **normalized** space, so under heterogeneous
variance high-σ categories dominate the shared gradient and are learned faster while
low-σ categories lag. Normalized-space loss removes this.

> Job ledger (which Slurm job was which) in [`00-experiments-log.md`](00-experiments-log.md).
> Canonical runs: SGD 30k = **12638** (v5, dense early eval), Adam 30k = 12583, SGD 100k = 12568.

## Setup

- Data: three sine categories, shared mean = 1, **distinct** freq/phase (so they stay
  distinguishable after instance normalization), amplitude scales giving variances
  **1 : 10 : 100** (`var1`, `var10`, `var100`). Variance is purely amplitude-driven.
- Model: tiny univariate patch transformer with RevIN-style instance norm
  (`a = mean`, `b = std` of the context); predicts the horizon in normalized space.
- Optimizer: **SGD** on purpose — it steps along the raw σ²-weighted gradient, so the
  bias is visible directly. Adam's per-coordinate RMS normalization divides out the
  *global* magnitude inflation (the same thing the `grad_norm_match` control isolates),
  so it is expected to *reduce* — not necessarily erase — the disparity. Switchable via
  `train.optimizer: {sgd, adam}`; the Adam variant is tracked below.
- Identical seed / weight init / stratified batch schedule across all runs; the **only**
  difference is the loss space (and the control toggles).
- Metric: per-category **nMSE in normalized space** (common metric for both runs), plus
  **global nMSE** (averaged over all categories). Measured on a **fixed held-out
  validation set** (`val_windows_per_category: 512` windows from unseen-phase series,
  constant `VAL_SEED` → identical across every setup/seed), not the training minibatch —
  so the curves are smooth, low-variance estimates rather than 8-window point estimates.
- **30k steps**, **5 seeds** per setup (`seeds: [0..4]`); curves show **mean ± 1 SD**
  shaded bands (`plot.band: std`, switch to `se`). Bands and the mean line are computed
  in **log space** (geometric mean ± log-std) since the axis is log and the per-seed
  spread crosses orders of magnitude — a linear band would clip below zero and paint a
  spurious full-height fill. Each convergence figure is saved on a full-range **log** axis
  plus **linear** companions zoomed to the early steps (`plot.linear_xlim: [200, 500, 2000]`
  → `*_linear_{200,500,2000}.png`, screen dir), since most convergence is in the first ~1k.
- Qualitative forecasts snapshotted on a non-uniform schedule (every 200 steps to 1k,
  every 1k to 10k, every 10k after); the static evolution figure shows the fixed columns
  `forecast_columns: [0, 100, 200, 500, 1000, 30000]` to visualize the disparate *rate*.
- W&B: runs grouped per setup as `{experiment}/{label}` (all 5 seeds in one group),
  tagged with `experiment / label / loss-mode / seed` (bump `wandb.experiment` to start a
  clean regeneration batch). Figures saved as both `.png` (preview) and `.pdf` (paper).

## Runs

| run | dataset | loss space | control |
|-----|---------|-----------|---------|
| `normalized` | heterogeneous | normalized | — |
| `original` | heterogeneous | original | — |
| `original_equalvar` | equal variance | original | scales collapsed to 1 |
| `original_gradmatch` | heterogeneous | original | grad rescaled to unit global norm |

## Hypotheses / falsifiable checks

1. `normalized`: per-category nMSE converges at ~equal pace (small spread).
2. `original`: graded by variance — `var100` fastest, `var1` slowest.
3. Step-0 per-category gradient magnitude under `original` follows the σ² ordering
   (≈ 1 : 10 : 100).
4. `original_equalvar`: disparity vanishes (≈ as flat as `normalized`).
5. `original_gradmatch`: disparity **persists** → it is loss-space bias, not a global
   learning-rate artifact.

## Results (lr=1e-3, 30k steps, GPU/Slurm, SGD)

5-seed GPU run (job **12638**, v5 dense early-eval schedule, held-out val metric, mean over
seeds; figures show mean ± 1 SD). Figures in `outputs/12638_loss_space_toy/`, metrics in
`summary.json`, live curves in W&B (`ts-normalization/loss_space_comparison`, group
`var_1_10_100_30k_v5/*`). All five checks pass, and the numbers reproduce the prior v3 run
(job 12582: spreads 4.05 / 296 / 5.98 / 80) almost exactly — the only v5 change was the
dense early eval cadence (every 5 steps to 200), so the curves are smooth at the start while
the final numbers are unchanged. The effect is stable, not an estimation artifact.

**nMSE spread (max/min across categories) and global nMSE:**

| run | spread | global nMSE | reading |
|-----|--------|-------------|---------|
| `normalized` | **4.09** | 5.0e-4 | baseline: task-difficulty only (different freqs), ~flat in variance |
| `original` | **292** | 2.7e-4 | variance bias dominates: `var100` reaches 2.7e-6, `var1` only 8.0e-4 |
| `original_equalvar` | **6.01** | 1.5e-3 | ≈ baseline → disparity was caused by variance (check 4 ✅) |
| `original_gradmatch` | **71** | 1.4e-4 | ≫ baseline → persists after matching global grad norm (check 5 ✅) |

Final per-category nMSE `[var1, var10, var100]`:
- `normalized`: `[8.2e-4, 2.0e-4, 4.9e-4]` — clustered; ordered by frequency, not variance.
- `original`:   `[8.1e-4, 2.1e-5, 2.7e-6]` — monotone in variance; `var100` ~300× better
  than `var1`, which descends at roughly the un-boosted (`σ²≈1`) rate.

**Steps to floor (first step with nMSE < 0.1, 5-seed mean).** A timescale read of the same
effect (cf. the Adam note's early-window table):

| source | SGD original | SGD normalized |
|--------|--------------|----------------|
| `var100` | 5   | 160 |
| `var10`  | 15  | 160 |
| `var1`   | 250 | 180 |

Original-space SGD spans 5→250 steps in σ² order — a ~50× time-to-threshold gap; normalized
space pulls all three into a ~20-step window (160–180). (Under Adam the same original-space
ordering holds, compressed: 10/20/105 — see [[adam-bias-early-window]].)

**Step-0 gradient magnitude ratio (original):** `1 : 9.54 : 94.0` ≈ σ² `1 : 10 : 100` —
direct confirmation of the Corollary that original-space scales the gradient by b²=σ²
(`grad_magnitude.png`: normalized bars flat, original bars climb as σ²).

**Convergence is rate-limited, not a wall.** At 1500 steps `var1` plateaued near 2e-2,
which looked like a floor. Training to 30k steps drives it to ~8e-4 (both spaces), ~20×
lower — so `var1` *does* converge, just slowly. The disparity is about convergence
**rate** under a fixed budget (the multi-dataset-pretraining reality), not the asymptote:
in original space `var100` reaches ~1e-5 within ~2k steps while `var1` needs the full 30k
to reach 8e-4. The `nmse_core` curves now trend down throughout with no flat tail.

> An earlier extreme spread (variances 1:100:10⁴, amplitudes 1:10:100) drove `original`
> to **NaN**: original-space SGD effectively uses `lr·σ²` and diverges on the
> high-variance category. That divergence is itself a manifestation of the bias; the
> 1:10:100 spread (amplitudes 1:3.16:10) stays stable while keeping the graded pace clear.

## 100k-step variant (job 12568)

Longer-budget rerun (`train.steps=100000`, experiment tag `var_1_10_100_100k_v2`, held-out
val metric, figures in `outputs/12568_loss_space_toy/`) to test whether the disparity is a
fixed-budget artifact — i.e. whether `var1` keeps descending or the original-space gap stays
open. **Verdict: the gap stays open.**

| setup | spread @30k (12582) | spread @100k (12568) |
|-------|---------------------|----------------------|
| `normalized` | 4.05 | 2.38 |
| `original` | **296** | **264** |
| `original_equalvar` | 5.98 | 2.72 |
| `original_gradmatch` | 80 | 29 |

`original` final per-category nMSE `[var1, var10, var100]`:
- @30k:  `[8.1e-4, 2.1e-5, 2.7e-6]`
- @100k: `[1.5e-4, 4.5e-6, 5.7e-7]`

**Everything keeps descending — no plateau.** `var1` drops 8.1e-4 → 1.5e-4 (~5×) with the
extra budget, confirming it is *rate-limited, not asymptote-limited*. But it **never catches
up**: `var10`/`var100` keep descending at their σ²-boosted rates too (`var100` 2.7e-6 →
5.7e-7), so the original-space spread is essentially unchanged (296 → 264) — `var1` stays
~260× behind throughout. Quadrupling the budget does not close the loss-space gap; the σ²
gradient weighting biases the *rate* at every point in training, not just early on.

The control/normalized spreads shrink with budget (the easy frequency-difficulty differences
wash out as all categories approach their floor), which makes the *persistence* of the
`original` spread the sharper contrast: it is structural, not budget-erasable.

> 12568 predates the per-window linear zooms and the `metrics_*.npz` persistence, so its
> linear companion is the un-zoomed full-range one and it cannot be replotted without a
> rerun. The log `nmse_core` (the figure that answers the var1-descent question) is complete.

## Figures

**Core comparison — `nmse_core.png`.** Left: normalized-space loss keeps all categories
in a tight bundle (equal pace). Right: original-space loss fans out, σ-ordered —
`var100` plunges ~2–3 orders of magnitude below `var1`. All curves trend down through
30k steps (no flat tail); the difference is the *rate*. Shaded bands are ±1 SD over seeds.

![nmse core](../outputs/12638_loss_space_toy/nmse_core.png)

**Controls — `nmse_controls.png`.** Left (equal variance): the fan-out collapses back to
the normalized baseline → the disparity was *variance*, not the frequency confound.
Right (grad-norm matched): the staggered fan-out persists even with global step size
equalized → loss-space bias, not a learning-rate artifact.

![nmse controls](../outputs/12638_loss_space_toy/nmse_controls.png)

**Global convergence — `nmse_global.png`.** Global nMSE (averaged over all categories)
per setup, mean ± 1 SD over seeds — shows each setup converging overall while the
per-category fan-out (above) is what differs.

![nmse global](../outputs/12638_loss_space_toy/nmse_global.png)

**Gradient scaling — `grad_magnitude.png`.** Per-category gradient magnitude at init:
normalized bars are flat, original bars climb as σ² (≈ 1 : 10 : 100), the direct
signature of the `b²=σ²` factor.

![grad magnitude](../outputs/12638_loss_space_toy/grad_magnitude.png)

**Qualitative forecasts — `forecasts_original.png`.** The original-space-trained model
fits `var10`/`var100` near-perfectly while under-serving `var1`, which barely received
gradient signal.

![forecasts original](../outputs/12638_loss_space_toy/forecasts_original.png)

For contrast, the normalized-space-trained model serves all three categories evenly:

![forecasts normalized](../outputs/12638_loss_space_toy/forecasts_normalized.png)

**Forecast evolution — `forecast_evolution_original.png`.** Small-multiples grid (rows =
category, columns = selected training steps `[0, 100, 200, 500, 1000, 30000]`); each cell
zooms into the forecast — target (black) vs prediction (red). Read *down* an early column:
by step ~200 `var10`/`var100` already match the target while `var1` is still far off — the
disparate rate in one glance. Read *across* a row to see that category converge.

![forecast evolution original](../outputs/12638_loss_space_toy/forecast_evolution_original.png)

Under normalized-space loss all three converge on a similar timescale:

![forecast evolution normalized](../outputs/12638_loss_space_toy/forecast_evolution_normalized.png)

## Reproduce

```sh
sbatch scripts/run.sbatch                         # GPU, wandb online
uv run python main.py wandb.mode=disabled         # local CPU, no wandb
```

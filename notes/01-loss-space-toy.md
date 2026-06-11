# 01 — Loss-space comparison toy

Demonstrates the §3 claim: computing the training loss in the **original** space scales
the gradient by `b^p = σ^p` relative to the **normalized** space, so under heterogeneous
variance high-σ categories dominate the shared gradient and are learned faster while
low-σ categories lag. Normalized-space loss removes this.

## Setup

- Data: three sine categories, shared mean = 1, **distinct** freq/phase (so they stay
  distinguishable after instance normalization), amplitude scales giving variances
  **1 : 10 : 100** (`var1`, `var10`, `var100`). Variance is purely amplitude-driven.
- Model: tiny univariate patch transformer with RevIN-style instance norm
  (`a = mean`, `b = std` of the context); predicts the horizon in normalized space.
- Optimizer: **SGD** on purpose — Adam is per-coordinate scale-invariant and would erase
  the σ² effect we are trying to expose.
- Identical seed / weight init / stratified batch schedule across all runs; the **only**
  difference is the loss space (and the control toggles).
- Metric: per-category **nMSE in normalized space** (common metric for both runs), plus
  **global nMSE** (averaged over all categories) to show overall convergence.
- **30k steps**, **5 seeds** per setup (`seeds: [0..4]`); curves show **mean ± 1 SD**
  shaded bands (`plot.band: std`, switch to `se` for standard error).
- Qualitative forecasts snapshotted on a non-uniform schedule (every 200 steps to 1k,
  every 1k to 10k, every 10k after) to visualize the disparate *rate* of convergence.
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

5-seed GPU run (job 12507, mean over seeds; figures show mean ± 1 SD). Figures in
`outputs/loss_space_toy/`, metrics in `summary.json`, live curves in W&B
(`ts-normalization/loss_space_comparison`, group `var_1_10_100/*`). All five checks pass.

**nMSE spread (max/min across categories) and global nMSE:**

| run | spread | global nMSE | reading |
|-----|--------|-------------|---------|
| `normalized` | **3.65** | 5.0e-4 | baseline: task-difficulty only (different freqs), ~flat in variance |
| `original` | **261** | 2.8e-4 | variance bias dominates: `var100` reaches 3.1e-6, `var1` only 8.2e-4 |
| `original_equalvar` | **5.58** | 1.5e-3 | ≈ baseline → disparity was caused by variance (check 4 ✅) |
| `original_gradmatch` | **79** | 1.4e-4 | ≫ baseline → persists after matching global grad norm (check 5 ✅) |

Final per-category nMSE `[var1, var10, var100]`:
- `normalized`: `[8.0e-4, 2.2e-4, 4.9e-4]` — clustered; ordered by frequency, not variance.
- `original`:   `[8.2e-4, 2.1e-5, 3.1e-6]` — monotone in variance; `var100` ~260× better
  than `var1`, which descends at roughly the un-boosted (`σ²≈1`) rate.

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

## Figures

**Core comparison — `nmse_core.png`.** Left: normalized-space loss keeps all categories
in a tight bundle (equal pace). Right: original-space loss fans out, σ-ordered —
`var100` plunges ~2–3 orders of magnitude below `var1`. All curves trend down through
30k steps (no flat tail); the difference is the *rate*. Shaded bands are ±1 SD over seeds.

![nmse core](../outputs/loss_space_toy/nmse_core.png)

**Controls — `nmse_controls.png`.** Left (equal variance): the fan-out collapses back to
the normalized baseline → the disparity was *variance*, not the frequency confound.
Right (grad-norm matched): the staggered fan-out persists even with global step size
equalized → loss-space bias, not a learning-rate artifact.

![nmse controls](../outputs/loss_space_toy/nmse_controls.png)

**Global convergence — `nmse_global.png`.** Global nMSE (averaged over all categories)
per setup, mean ± 1 SD over seeds — shows each setup converging overall while the
per-category fan-out (above) is what differs.

![nmse global](../outputs/loss_space_toy/nmse_global.png)

**Gradient scaling — `grad_magnitude.png`.** Per-category gradient magnitude at init:
normalized bars are flat, original bars climb as σ² (≈ 1 : 10 : 100), the direct
signature of the `b²=σ²` factor.

![grad magnitude](../outputs/loss_space_toy/grad_magnitude.png)

**Qualitative forecasts — `forecasts_original.png`.** The original-space-trained model
fits `var10`/`var100` near-perfectly while under-serving `var1`, which barely received
gradient signal.

![forecasts original](../outputs/loss_space_toy/forecasts_original.png)

For contrast, the normalized-space-trained model serves all three categories evenly:

![forecasts normalized](../outputs/loss_space_toy/forecasts_normalized.png)

**Forecast evolution — `forecast_evolution_original.png`.** Probe forecast snapshotted
through training, colored light→dark by step (colorbar = training step). The disparate
*rate* is explicit: `var10`/`var100` lock onto the target within a few hundred steps
(early colors already overlay the black target), while `var1`'s predictions crawl up from
purple and only the latest (yellow) snapshots approach it.

![forecast evolution original](../outputs/loss_space_toy/forecast_evolution_original.png)

Under normalized-space loss all three converge on a similar timescale:

![forecast evolution normalized](../outputs/loss_space_toy/forecast_evolution_normalized.png)

## Reproduce

```sh
sbatch scripts/run.sbatch                         # GPU, wandb online
uv run python main.py wandb.mode=disabled         # local CPU, no wandb
```

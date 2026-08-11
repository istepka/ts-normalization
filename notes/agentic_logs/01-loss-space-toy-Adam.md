# 01 (Adam) — Loss-space comparison toy under Adam

Companion to [`01-loss-space-toy.md`](01-loss-space-toy.md) (the SGD run). **Identical**
setup, data, seeds, init, schedule, and lr — the *only* change is
`train.optimizer: sgd → adam`. The question: does Adam's per-coordinate RMS
normalization neutralize the original-space `b²=σ²` gradient bias that SGD exposes?

**Answer: no — it only *compresses and then hides* it.** The σ²-ordered convergence-**rate**
disparity is still present under Adam: in original space, steps to reach nMSE < 0.1 are
`var100`≈10, `var10`≈20, `var1`≈105 (vs normalized space, where all three hit it by ≈10) —
the same variance ordering SGD shows (5 / 15 / 250), compressed from a ~50× gap to ~10× but
not removed. The reason the **final-step** spread looks small
(3.5× vs SGD's 292×) is an artifact: by 30k all sources have saturated at the ~1e-5 noise
floor (the model has effectively memorized the 3-sine problem — we are **overfitting**), so
the max/min ratio there is dominated by jitter, not by the bias. Read at the endpoint the
bias looks gone; read in the first ~150 steps (the linear zoom) it is clearly there. This is
consistent with the unchanged init gradient ratio (1:9.54:94).

> **Correction (2026-06-11):** an earlier version of this note claimed Adam "almost entirely"
> neutralizes the bias, based on the final-step spread. That metric is measured deep in the
> saturated/overfit regime and understates the effect; the early-window linear curves are the
> honest read. See [[loss-space-experiment]].

## Setup

Everything as in the SGD note, except:
- Optimizer: **Adam** (`train.optimizer: adam`), lr = 1e-3 (same as SGD, for a clean
  one-variable comparison — not separately tuned for Adam).
- 30k steps, 5 seeds, mean ± 1 SD bands.
- Run: job 12646 (v5, dense early eval), experiment tag `var_1_10_100_30k_adam_v5`, figures
  in `outputs/12646_loss_space_toy/`, W&B group `var_1_10_100_30k_adam_v5/*`.

## Results — Adam vs SGD (30k steps, 5 seeds)

**Early-window convergence rate (the honest signal).** Steps to floor = first step with
nMSE < 0.1 (5-seed mean), original vs normalized space:

| source | Adam original | Adam normalized | (SGD original, for scale) |
|--------|---------------|-----------------|---------------------------|
| `var100` | 10  | 10 | 5 |
| `var10`  | 20  | 10 | 15 |
| `var1`   | 105 | 10 | 250 |

So original-space Adam still fans out by σ² — `var1` takes ~10× longer than `var100` — while
normalized-space Adam pulls all three to the floor together (~10 steps). Adam compresses the
gap relative to SGD (~50× → ~10×) but does **not** remove it: the σ² ordering and the
disparity survive. The grad-norm-matched control shows the same fan-out, so it is loss-space
bias, not a global step-size effect.

**Final-step nMSE spread (max/min) — measured in the saturated regime, understates the bias:**

Held-out val metric, 5 seeds. SGD = job 12638, Adam = job 12646 (both v5 dense early eval;
figures in `outputs/12646_loss_space_toy/`, W&B group `var_1_10_100_30k_adam_v5/*`).

| setup | SGD spread (job 12638) | Adam spread (job 12646) |
|-------|------------------------|-------------------------|
| `normalized`        | 4.09    | 2.77 |
| `original`          | **292** | 3.46 |
| `original_equalvar` | 6.01    | 1.45 |
| `original_gradmatch`| 71      | 2.97 |

The Adam column looks flat across setups, but that is **not** evidence the bias is gone — it
is because by 30k every source under Adam has bottomed out at the ~1e-5 floor, where the
max/min ratio is just relative-error jitter (see overfitting note below). The fan-out happened
and finished inside the first ~150 steps; this endpoint metric is taken long after. The SGD
column, by contrast, is still mid-convergence at 30k, so its spread is meaningful. **Do not
compare the two columns as if they measured the same thing** — they are read at different
points along very different timescales.

**Global nMSE (averaged over categories):**

| setup | SGD global | Adam global |
|-------|-----------|-------------|
| `normalized`        | 5.0e-4 | 4.2e-5 |
| `original`          | 2.7e-4 | 2.3e-5 |
| `original_equalvar` | 1.5e-3 | 1.0e-5 |
| `original_gradmatch`| 1.4e-4 | 3.9e-5 |

All Adam endpoints sit near the ~1e-5 floor, so these globals reflect the saturated regime,
not convergence rate.

Final per-category nMSE `[var1, var10, var100]` (all at the floor — endpoint only):
- Adam `normalized`: `[6.4e-5, 2.3e-5, 3.8e-5]` — flat.
- Adam `original`:   `[3.7e-5, 2.2e-5, 1.1e-5]` — still variance-ordered (var100 lowest), but
  compressed to 3.5× because all three have bottomed out; the *ordering* survives, the
  *magnitude of the gap* is washed out by saturation. The rate gap lived in steps 0–150.

## What Adam actually does (it speeds convergence, it does not remove the bias)

The **step-0 gradient ratio is identical to the SGD run**: `1 : 9.54 : 94 ≈ σ² (1:10:100)`.
The optimizer does not change the gradient — the `b²=σ²` weighting of each category's
contribution is fully present in `∂L/∂θ` either way.

Adam divides each coordinate by the running RMS of its own gradient (`update = g/√v`). This
removes the *global* magnitude inflation (so original-space Adam does not diverge the way
SGD would at extreme spreads, and it converges much faster overall), but it does **not**
equalize the per-source convergence rate: in original space `var100` still reaches the floor
first and `var1` last, in σ² order. The diagonal preconditioner rescales coordinate-wise
magnitude, not the variance-driven *direction/timing* of which source dominates the shared
parameters early on. The grad-norm-matched control fanning out the same way confirms it is a
loss-space effect, not a global step-size one. Net: Adam compresses the σ² fan-out into a
short early window and then the model overfits, so a late metric cannot see it.

## Why the Adam curves are noisy (expected, not a bug)

The curves bounce 2–3 orders on the log axis with wide SD bands. This is **not** an
estimation artifact (the val set is large and fixed); it is genuine and expected for
**constant-lr Adam at a very low loss floor**:

- Adam's update is `g/√v`, normalized to ~`lr` magnitude per coordinate regardless of
  gradient size. SGD's steps shrink as gradients shrink near the optimum, so SGD settles;
  Adam keeps taking ~`lr`-sized steps and jitters in a ball of radius ~`lr` around the
  minimum — it never stops moving.
- The model reaches near-perfect fit (nMSE ~1e-6), where relative error is hypersensitive:
  a tiny `lr=1e-3` weight perturbation swings nMSE from ~1e-6 to ~1e-3 even though the
  absolute prediction barely moves. The **log** axis turns that into a 2–3 order bounce
  (on a linear axis it is a flat near-zero wiggle).
- Those oscillations are out of phase across the 5 seeds, so the across-seed SD is large.
- Tiny batch (24) feeds extra gradient noise into both the step and Adam's `v`.

Decision: **keep constant lr and accept the noise** — it does not affect the conclusion (the
rate disparity is read in the early window, before the noise floor matters). A warmup→cosine
schedule or lower lr would let Adam settle and clean the late curves, but that is a
methodology change we are deliberately not making here.

## Overfitting — why the endpoint is meaningless

This is a tiny problem (3 deterministic sine sources, no observation noise) and a transformer
with far more capacity than needed, trained for 30k steps. Adam drives every source to nMSE
~1e-5 within a few hundred steps and then keeps fitting — it has essentially **memorized**
the sources. Past ~200 steps the per-source nMSE is at the relative-error noise floor, where
differences between sources (and between seeds) are jitter, not signal. So any metric read at
30k — the final spread, the late curve shape — says almost nothing about the loss-space bias.
The bias is a statement about *convergence rate*, and the only place it is legible is early
training (steps 0–~150), which is exactly what the linear-zoom figures show. The log figures
spend most of their x-range plotting this meaningless saturated tail, which is what made the
effect look absent under Adam.

## Implication for the paper

Real TSFMs train with Adam/AdamW, so the honest message is: **the σ²-driven per-source
convergence-rate disparity persists under Adam** (it is visible in the first ~150 steps and
in the unchanged init grad ratio 1:9.5:94); Adam does not remove it, it only accelerates
convergence so the fan-out finishes early and is then buried under overfitting. The earlier
"Adam fixes it" reading was an artifact of measuring a saturated endpoint. Caveats to state:
- This toy overfits hard (deterministic data, over-capacity model), so the *late* dynamics
  are uninformative; the claim rests on early-training rate, which is where pretraining of a
  real TSFM on finite data also spends its informative budget.
- A real run never reaches a 1e-5 floor on every source, so the disparity would not get
  "hidden" the same way — the early-training rate gap is the regime that matters at scale.
- Worth probing: does the gap *widen* with larger variance spread, longer-tailed source
  mixes, or parameter sharing across very heterogeneous scales where a diagonal
  preconditioner cannot separate sources?

## Figures

**Read the LINEAR early-window zoom, not the log full-range figure.** The log `nmse_core`
spends most of its x-range on the saturated tail and makes the bias look absent; the linear
zoom over the first 200 steps is where the fan-out is legible.

**Core comparison (linear, first 200 steps) — `nmse_core_linear_200.png`.** Left
(normalized): all three sources crash together by ~step 25. Right (original): clear σ²
fan-out — `var100` (green) reaches the floor by ~step 5, `var10` (orange) by ~25, `var1`
(blue) only by ~125. The same staggering SGD shows over 30k steps, here compressed into ~150.

![nmse core adam linear](../outputs/12646_loss_space_toy/nmse_core_linear_200.png)

**Controls (linear, first 200 steps) — `nmse_controls_linear_200.png`.** Left (equal
variance): bundled, no fan-out → the disparity was variance. Right (grad-norm matched): the
σ² fan-out persists (`var1` ~60 vs `var100` ~5) → loss-space bias, not a global step-size
artifact, under Adam too.

![nmse controls adam linear](../outputs/12646_loss_space_toy/nmse_controls_linear_200.png)

**Full-range log — `nmse_core.png` (kept for reference, but misleading on its own).** Both
panels look bundled at ~1e-5 because the plot is dominated by the post-convergence saturated
tail where everything has overfit; the fan-out is squeezed into the first few pixels.

![nmse core adam](../outputs/12646_loss_space_toy/nmse_core.png)

**Gradient scaling — `grad_magnitude.png`.** Unchanged from SGD: original-space init
gradients climb as σ² (≈ 1 : 10 : 100). The bias lives in the gradient and survives into the
early-training rate; Adam rescales coordinate magnitude but not which source converges first.

![grad magnitude adam](../outputs/12646_loss_space_toy/grad_magnitude.png)

## Reproduce

```sh
sbatch --job-name=loss_space_adam scripts/run.sbatch \
  train.optimizer=adam wandb.experiment=var_1_10_100_30k_adam
```

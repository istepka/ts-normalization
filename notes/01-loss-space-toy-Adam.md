# 01 (Adam) — Loss-space comparison toy under Adam

Companion to [`01-loss-space-toy.md`](01-loss-space-toy.md) (the SGD run). **Identical**
setup, data, seeds, init, schedule, and lr — the *only* change is
`train.optimizer: sgd → adam`. The question: does Adam's per-coordinate RMS
normalization neutralize the original-space `b²=σ²` gradient bias that SGD exposes?

**Answer: yes, almost entirely.** Under Adam the original-space per-category disparity
collapses from SGD's **261×** down to **3.2×** — the same range as the normalized and
equal-variance baselines. The bias is still present in the raw gradient (the init
gradient ratio is unchanged), but Adam divides it out of the *update*, so all categories
converge at ~equal pace regardless of loss space.

## Setup

Everything as in the SGD note, except:
- Optimizer: **Adam** (`train.optimizer: adam`), lr = 1e-3 (same as SGD, for a clean
  one-variable comparison — not separately tuned for Adam).
- 30k steps, 5 seeds, mean ± 1 SD bands.
- Run: job 12547, experiment tag `var_1_10_100_30k_adam`, figures in
  `outputs/12547_loss_space_toy/`, W&B group `var_1_10_100_30k_adam/*`.

## Results — Adam vs SGD (30k steps, 5 seeds)

**Per-category nMSE spread (max/min across categories) — the bias signal:**

| setup | SGD spread (job 12521) | **Adam spread (job 12547)** |
|-------|------------------------|-----------------------------|
| `normalized`        | 3.65    | **1.30** |
| `original`          | **261** | **3.17** |
| `original_equalvar` | 5.58    | **4.16** |
| `original_gradmatch`| 79      | **2.07** |

Under Adam every setup sits in the **1.3–4.2×** band: the variance-driven fan-out is
gone. The original-space spread (3.17) is essentially the normalized baseline (1.30) —
versus SGD where original (261) was ~71× its own normalized baseline (3.65).

**Global nMSE (averaged over categories):**

| setup | SGD global | Adam global |
|-------|-----------|-------------|
| `normalized`        | 5.0e-4 | 4.0e-6 |
| `original`          | 2.8e-4 | 2.9e-5 |
| `original_equalvar` | 1.5e-3 | 7.9e-6 |
| `original_gradmatch`| 1.4e-4 | 5.3e-5 |

Adam also converges ~50–100× lower overall within the same 30k budget (adaptive steps),
as expected.

Final per-category nMSE `[var1, var10, var100]`:
- Adam `normalized`: `[3.7e-6, 3.6e-6, 4.7e-6]` — flat.
- Adam `original`:   `[4.2e-5, 3.2e-5, 1.3e-5]` — still *mildly* variance-ordered
  (var100 lowest), but only 3.2× across the whole range vs SGD's 261×.

## Why the bias vanishes under Adam (but is still in the gradient)

The **step-0 gradient ratio is identical to the SGD run**: `1 : 9.54 : 94 ≈ σ² (1:10:100)`.
The optimizer does not change the gradient — the `b²=σ²` weighting of each category's
contribution is fully present in `∂L/∂θ` either way.

What changes is the *update*. SGD steps along that raw σ²-weighted gradient, so var100
dominates the shared parameters and converges first. Adam divides each coordinate by the
running RMS of its own gradient (`update = g/√v`), which removes the per-coordinate
magnitude inflation — and with it the global b² scaling — so the categories advance at
~equal pace. `original_gradmatch` collapsing too (79 → 2.07) is consistent: once Adam
normalizes per coordinate, an extra global-norm match adds almost nothing.

## Implication for the paper

Real TSFMs train with Adam/AdamW, and this toy says **Adam largely neutralizes the
loss-space gradient bias** — an important caveat the §3 claim should address head-on. The
honest framing: the bias is real and provable *in the gradient* (init ratio 1:9.5:94,
unchanged), but the realistic optimizer's diagonal preconditioner masks it in this
controlled toy. Open questions worth probing before claiming practical impact at scale:
- Does it resurface in regimes where Adam's per-coordinate normalization is imperfect —
  warmup, large `ε`, low-precision second moments, parameter sharing across very
  heterogeneous scales where a diagonal preconditioner cannot separate categories?
- Is the residual mild ordering (3.2×) the visible tail of a bias that grows with the
  variance spread or with model/parameter coupling?

## Figures

**Core comparison — `nmse_core.png`.** Both panels (normalized *and* original) keep all
three categories bundled, descending together to ~1e-5 with no variance-ordered fan-out —
the visual opposite of the SGD original-space panel. (Curves are spiky: Adam at lr=1e-3 on
this tiny problem takes noisy steps, but there is no systematic σ-separation.)

![nmse core adam](../outputs/12547_loss_space_toy/nmse_core.png)

**Controls — `nmse_controls.png`.** Equal-variance and grad-norm-matched both look like the
core panels under Adam — flat, no fan-out — confirming the disparity is gone across the board.

![nmse controls adam](../outputs/12547_loss_space_toy/nmse_controls.png)

**Gradient scaling — `grad_magnitude.png`.** Unchanged from SGD: original-space init
gradients still climb as σ² (≈ 1 : 10 : 100). The bias lives in the gradient; Adam just
removes it from the update.

![grad magnitude adam](../outputs/12547_loss_space_toy/grad_magnitude.png)

## Reproduce

```sh
sbatch --job-name=loss_space_adam scripts/run.sbatch \
  train.optimizer=adam wandb.experiment=var_1_10_100_30k_adam
```

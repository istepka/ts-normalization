# Scale-free evaluation metrics for the GiftEvalPretrain loss-space study

Date: 2026-08-08

## The defect

The 12 GiftEvalPretrain runs reported `final_pooled_mse` and per-dataset Gini
computed on **raw MSE in whichever space each variant trained in**:

- `run_moment` scored on `strat_out.per_example_loss_masked` (train.py)
- `run_timesfm` scored on `strat_out.mse_per_example`

`per_example_loss_masked` is the training objective, selected by condition
(`moment_adapter.forward` picks `normalized_mse` for `moment_normalized` and
`original_mse` for `moment_original`). So the normalized variant was scored in
normalized space and the original variant in original space, and those two numbers
were then tabulated side by side and compared. The variants were never on a common
metric.

Two consequences:

1. **Cross-variant comparisons were meaningless.** Comparing a normalized-space
   MSE against an original-space MSE compares a dimensionless quantity to one
   in squared physical units.
2. **Gini measured units, not model quality.** Gini over raw per-dataset MSE
   ranks datasets by the magnitude of their numbers. `fred_md` (FRED
   macroeconomic series, values in the billions) sat at 5.85e12 while the next
   dataset was ~300x smaller, which alone forces Gini to ~0.97 regardless of
   how well the model fits anything. Every condition reported Gini in the
   0.96-0.98 band, which is what a units artifact looks like.

The huge `final_pooled_mse` values (587M for `timesfm_native_original_A`) were
the visible symptom, not the disease: pooled MSE is an unweighted mean over
windows, so one series in the trillions swamps several hundred other windows.

Two smaller defects surfaced while fixing this:

3. **Eval ran with dropout active.** Neither `run_moment` nor `run_timesfm`
   called `model.eval()` around the checkpoint eval block, so MOMENT
   (dropout 0.1) sampled dropout masks at evaluation time, adding noise to
   every reported metric. Both now toggle `eval()`/`train()` around the block.
4. **Frequency handling was too narrow.** The first cut of the MASE seasonal
   period was a literal lookup table of alias strings, which would have
   raised on compound aliases (`W-SUN`, `A-DEC`) that `frequency_bucket`
   already handled. It now derives the period from the parsed offset's
   duration, and the offset parser is shared rather than duplicated.

## Why raw MSE was the wrong target in the first place

`overleaf/sections/results.tex` already settled this convention. The synthetic
loss-space study reports per-source **nMSE** and computes Gini on per-source
nMSE (`tab:loss-space-synthetic-gini`). The real-data scale-swap study
summarizes convergence by **nMSE AUC**. The GiftEval TSFM experiment was the
only one reporting raw MSE, and nobody evaluates TSFMs on aggregated raw MSE
across datasets with different units.

Critically, a scale-free metric does **not** wash out the scale effect. The
treatment in Theorem `theorem:norm-bias` is that original-space loss scales the
**gradient** by `b^p`, which produces a genuinely different trained model.
Evaluating that model on a scale-free metric is exactly what isolates the
effect. `results.tex` line 193 confirms the design: the normalized-space
endpoints for `b=1` and `b=10` are nearly identical (the control, showing the
metric is scale-free), while the original-space condition still separates.

## The fix

**Metric primitives** (`losses.py`):

- `masked_mae`, mirroring `masked_mse`'s masking contract, as the MASE
  numerator.
- `SEASONAL_PERIODS`, a freq -> seasonal lag table following the
  GIFT-Eval / gluonts `get_seasonality` convention. Covers every frequency in
  the corpus (H, 5T, D, 4S, 30T, 15T, M, T, 10T, W).
- `parse_offset` / `seasonal_period`, deriving the MASE lag from the parsed
  offset's duration (sub-daily takes the daily cycle, daily takes weekly,
  monthly 12, quarterly 4, weekly and yearly 1). Handles multipliers and
  anchors without enumerating spellings. `timesfm_model.frequency_bucket` now
  shares this parser instead of keeping its own copy.
- `seasonal_naive_mae`, the in-sample seasonal-naive MAE denominator
  `mean|y_t - y_{t-m}|` over the context window. Near-constant windows
  (denominator below 1e-8) return NaN and are dropped, rather than emitting a
  divide-by-near-zero MASE that would dominate any mean or Gini far worse
  than raw MSE ever did.

  When `m` does not fit the context the lag falls back to 1 (random-walk
  naive). This matters for real datasets: `4S` implies a 21,600-step daily
  cycle against a 512-step context, so a strict rule would drop that dataset
  entirely. Dropping whole datasets distorts a dispersion metric more than a
  shorter lag does, and since every window of a dataset shares its frequency,
  the lag is constant within a source either way.
- `group_mean_by_source`, `pooled_mean`, and `dispersion_metrics` are now
  NaN-tolerant so dropped windows and fully-dropped sources aggregate cleanly.

**Adapters**: both now expose condition-independent evaluation metrics
alongside the training objective. `MomentForwardResult` and
`TimesFMForwardResult` gained `mase`; `TimesFMForwardResult` also gained
`normalized_mse` (MOMENT already had it). The training loss is untouched, so
the treatment is preserved exactly.

**Eval call sites** (`train.py`): both `run_moment` and `run_timesfm` now score
on `normalized_mse` (primary) and `mase` (secondary) regardless of condition.
`history.json`'s top-level `dataset`/`domain`/`frequency` breakdown is now
nMSE, which keeps `aggregate_tsfm_loss_space.py` working unchanged while making
what it reads correct; MASE lands in a nested `"mase"` key.

## Metric definitions, and why there is no separate "nMASE"

- **nMSE**: MSE between the normalized reconstruction/forecast and the
  normalized target, i.e. `src/loss.py:per_sample_nmse`'s definition, which is
  what `results.tex` reports throughout.
- **MASE**: masked original-space MAE divided by the in-sample seasonal-naive
  MAE of the same window.

MASE computed in normalized space is *identical* to MASE in original space:
the affine normalization `(x - a) / b` divides numerator and denominator by
the same `b`, and the shift `a` cancels from both because each is built from
differences. So "nMASE" is not a distinct metric, and reporting it as a third
column would be reporting the MASE column twice. This is asserted directly in
`tests/test_losses.py::test_normalized_space_mase_equals_original_space_mase`
rather than left as a claim.

## nMSE is not robust on this corpus; MASE is

Rescoring `moment_original_B` on the corrected metrics gave pooled nMSE 1.27
and pooled MASE 1.88, both sane, but a per-dataset nMSE Gini of 0.976 and a
per-dataset nMSE *mean* of 24,477. A dimensionless quantity averaging 24,000
is a red flag, so it was diagnosed rather than reported
(`scripts/_diagnose_nmse.py`, one checkpoint, stratified eval sample):

```
nMSE  min=3.95e-02 med=9.73e-01 max=6.58e+07
windows with nMSE > 100: 5 / 2376
top window: m5  nmse=6.58e+07  ctx_std=4.42e-02
```

The median nMSE is 0.97, exactly where it should be. The entire inflation is
**one window**. `m5` is retail unit sales, mostly zeros with rare spikes; that
window's context standard deviation is 0.044, so RevIN divides by ~0.044 and a
single spike becomes a normalized value in the thousands. Its nMSE of 6.6e7,
spread over 64 windows, sets `m5`'s dataset mean to ~1.0e6, and that one
dataset then determines the corpus Gini.

This is the near-zero-sigma failure mode `loss_space_gradient_bias.tex` already
names ("the near-zero sigma failure mode remains"). It does not appear in the
synthetic study because there each source has a fixed, known sigma; it appears
here because RevIN's per-window sigma is data-dependent and can be far smaller
than a rare spike in the same window.

Consequences for the experiment design:

- **MASE is the primary per-dataset error and the primary Gini input.** It is
  scale-free, it is what TSFM benchmarks actually report, and its denominator
  is a seasonal-naive MAE over the whole context rather than a per-window
  standard deviation, so a single spike cannot collapse it. Its Gini on the
  same run is 0.341 with all 42 datasets contributing.
- **nMSE is retained but reported two ways**: the mean-reduced version (the
  paper's definition, kept for continuity) and a median-reduced version
  (`L.group_median_by_source`, `source_breakdown(..., reducer="median")`),
  which answers the same question without letting one window set the answer.
  Where the two disagree, the mean-reduced number is measuring an outlier.

The lesson generalizes: the original Gini of ~0.97 on raw MSE and this Gini of
0.976 on mean nMSE have the same shape of cause, a single extreme value
dominating an unweighted mean. Fixing the units was necessary but not
sufficient; the aggregation has to be robust too.

## Recomputing without retraining

Because the metric is a pure function of (model weights, eval windows) and
nothing about training changed, the corrected metrics are recoverable from the
saved checkpoints. `scripts/recompute_tsfm_scale_free_metrics.py` reloads every
`checkpoint_step*.pt` and rescores it.

One design decision: **every run is evaluated on natural, unscaled held-out
windows** regardless of the scale assignment it trained under, so all 12 variants
are scored on identical data. The controlled scale `b` is a training-time
intervention only.

Resolution limit: checkpoints were saved every 2,000 steps (15 per run), so
this recovers final-step Gini and a coarse trajectory, but not the 120-point
`eval_every=250` resolution needed for a clean AUC through step 2,000. A
re-run is required only if the AUC axis is needed at full resolution.

## Results

Final step (30,000), stratified eval sample, 42 datasets, every variant scored on
identical natural-scale windows with chunked dropout-free evaluation.

The six TimesFM variants below are the **retrained** ones (`gifteval_timesfm_fixed`,
job 26669) carrying the first-patch sigma fix. The six MOMENT variants are the
original runs, rescored offline; MOMENT was never affected by that fix. Numbers
live in `outputs/tsfm_scale_free_metrics_v2/`; the pre-fix set is preserved in
`outputs/tsfm_scale_free_metrics/`.

| condition | pooled MASE | MASE Gini | median-nMSE | med-nMSE Gini | mean-nMSE Gini |
|---|---|---|---|---|---|
| moment_normalized_natural | 0.667 | 0.163 | 0.325 | 0.458 | 0.9762 |
| moment_normalized_A | 0.672 | 0.162 | 0.328 | 0.452 | 0.9762 |
| moment_normalized_B | 0.670 | 0.166 | 0.325 | 0.455 | 0.9762 |
| moment_original_natural | 2.092 | 0.369 | 1.000 | 0.104 | 0.9761 |
| moment_original_A | 1.981 | 0.351 | 0.929 | 0.123 | 0.9761 |
| moment_original_B | 2.001 | 0.347 | 0.990 | 0.134 | 0.9761 |
| timesfm_normalized_natural | 2.242 | 0.407 | 6.540 | 0.542 | 0.9635 |
| timesfm_normalized_A | 2.395 | 0.394 | 6.302 | 0.441 | 0.9673 |
| timesfm_normalized_B | 2.083 | 0.407 | 5.644 | 0.513 | 0.9642 |
| timesfm_original_natural | 2.065 | 0.456 | 18.875 | 0.833 | 0.9613 |
| timesfm_original_A | 1.999 | 0.433 | 7.100 | 0.621 | 0.9637 |
| timesfm_original_B | 2.006 | 0.457 | 12.454 | 0.749 | 0.9603 |

### The headline: the effect is real for MOMENT and absent for TimesFM

Rank-based head-to-head on per-dataset error, normalized vs original. Because
both variants are scored on identical windows, the per-dataset MASE ratio has the
same seasonal-naive denominator top and bottom and therefore equals the raw
original-space MAE ratio exactly -- the baseline choice cannot bias it.

| model | setting | normalized wins (MASE) | median ratio | Wilcoxon p |
|---|---|---|---|---|
| MOMENT | natural | 42/42 | 0.414x | 4.6e-13 |
| MOMENT | A | 42/42 | 0.451x | 4.6e-13 |
| MOMENT | B | 42/42 | 0.418x | 4.6e-13 |
| TimesFM | natural | 11/42 | 1.090x | 0.070 |
| TimesFM | A | 7/42 | 1.217x | 6.5e-04 |
| TimesFM | B | 16/42 | 1.070x | 0.303 |

**MOMENT.** Normalized-space loss wins on every dataset, in every scale
setting, and the result is metric-independent: 42/42 also holds under nMSE
(median ratio 0.364x) and under median-nMSE (0.215x), which uses no baseline at
all. Pooled MASE 0.67 vs 2.09 and MASE Gini 0.163 vs 0.369 (2.3x lower
inequality, closely matching the synthetic study's 0.273 vs 0.645 = 2.4x).

**TimesFM.** The same comparison is a coin flip that mildly favours *original*
space. Median ratios sit in 0.89-1.22x across every setting and metric, i.e.
within +/-20%, and the MASE Gini gap shrinks to 0.407 vs 0.456 against MOMENT's
0.163 vs 0.369.

So the defensible claim is narrower than "normalized-space loss is better". It
is decisively better for MOMENT's masked-reconstruction objective and
neutral-to-slightly-worse for TimesFM's forecasting objective, on the same
corpus, window index, and schedule. The difference is in the objective, not the
data.

**Hypothesis, not tested here.** The likely mechanism is where the target sits
relative to the normalization statistics. MOMENT reconstructs positions
*inside* the window that RevIN normalized, so the target is always on the scale
the loss is measured in. TimesFM forecasts *beyond* the context using
first-patch statistics only, so the target routinely sits off that scale. That
is the same structural fact that makes the degenerate windows ill-posed (see
the TimesFM section below). Testing it would need a variant that normalizes
TimesFM by whole-context statistics.

**Mean-nMSE Gini is confirmed dead.** It is 0.976 for every MOMENT variant to four
decimals, because `m5`'s per-dataset nMSE is 1.028e6 in *both* the normalized
and original variants, i.e. essentially independent of the model. It measures one
window's normalization, not model quality. Do not report it.

**The median-nMSE Gini column inverts for MOMENT, and it is a trap.** It reads
0.10-0.13 for the original variant against 0.45 for the normalized variant, apparently
favouring original-space. But the original variant's median nMSE is ~1.00 on
essentially every dataset, and nMSE = 1 is exactly the trivial predictor
(predicting the normalized mean, i.e. zero). The original variant is uniformly *at
the failure floor*, and uniform failure has low dispersion. Low Gini there
measures equality of failure, not equity. Gini must be read next to the level
of the metric, never alone.

### Paired scale-assignment effect (A - B), per dataset, final step

| pair | metric | mean effect | 95% CI | p | favours B |
|---|---|---|---|---|---|
| moment_normalized | MASE | -0.0015 | 0.0121 | 0.805 | 26/42 |
| moment_normalized | median-nMSE | +0.0027 | 0.0036 | 0.128 | 26/42 |
| moment_original | MASE | -0.1649 | 0.1892 | 0.086 | 8/42 |
| moment_original | median-nMSE | -0.0612 | 0.0274 | **0.0001** | 8/42 |
| timesfm_normalized | MASE | +0.3800 | 0.2121 | **0.0008** | 38/42 |
| timesfm_normalized | median-nMSE | +0.6586 | 0.9853 | 0.184 | 35/42 |
| timesfm_original | MASE | -0.1717 | 0.1595 | **0.036** | 18/42 |
| timesfm_original | median-nMSE | -5.3534 | 8.6368 | 0.218 | 11/42 |

For MOMENT this is the predicted pattern: under normalized-space loss the scale
assignment does nothing (p = 0.81 / 0.13, a 26/42 coin flip), which is the
control confirming the metric is scale-free; under original-space loss the
assignment matters (p = 0.0001 on median-nMSE, 34 of 42 datasets favouring
assignment A). That is Theorem `theorem:norm-bias` on real data through a
scale-free metric.

**Unexplained, do not report yet.** `timesfm_normalized` shows a significant
MASE effect (p = 0.0008, 38/42 favouring B). A normalized-space variant is supposed
to be the scale-invariant control, and `moment_normalized` behaves that way
(p = 0.81). Either TimesFM's first-patch normalization does not fully remove
the imposed scale, or something else is leaking. This needs diagnosis before it
appears anywhere.

Direction caveat for all rows: A and B are complementary assignments, so "A
better" is a statement about which datasets drew `b = 10` under A, not a direct
claim that `b = 10` helps. Mapping the effect onto `b` requires joining
per-dataset effects to `window_index.scale_for`, not done here.

## The TimesFM fix, and why it goes on our side of the vendor boundary

Diagnosed on real training batches from a fresh init
(`scripts/_diagnose_timesfm.py`):

```
n windows: 2048
sigma at clamp floor (<=1e-6): 70
|normalized target| max: med=4.28e+00  max=1.12e+09
normalized MSE:        med=3.65e+00  max=3.20e+17
top 1% of windows contribute 100.0% of total normalized-space loss
```

The mechanism is not the one a context-wide filter would catch.
`_masked_mean_std` states its own contract: *"We return the statistics of the
first patch with more than three non-padded values."* So mu and sigma come
from the **first 32 timesteps**, not the 512-step context, and sigma is then
clamped to `config.tolerance = 1e-6`. A series that merely *starts* flat gets
sigma = 1e-6 even when the rest of its context varies. Filtering on the window
index's `context_std` flags only 6,609 of 41M windows (0.016%) and is
therefore the wrong test -- it measures a different statistic than the one the
model divides by.

Crucially, **this is not a defect in TimesFM**. The released objective is
computed after `_reverse_transform`, where sigma cancels, so the clamp is
harmless and `timesfm_native_original` is healthy and internally consistent
(pooled MASE 2.24 / 2.36 / 2.26 across its three variants). The
normalized-space condition is *this study's* addition, and it is what becomes
ill-posed: it divides the target by that same first-patch sigma.

The fix therefore stays outside the vendored file, which remains byte-for-byte
at its pinned revision with the clamp and the first-patch convention intact.
`timesfm_model.forward` flags `degenerate = sigma <= tolerance * (1 + 1e-6)`,
using TimesFM's own statistic; `training_step_metrics` excludes those windows
from the loss, and both reported metrics return NaN for them so the existing
NaN-dropping aggregation removes them. `degenerate_frac` is logged per step.
Asserted in
`tests/test_timesfm_model.py::test_flat_first_patch_window_is_flagged_degenerate_and_excluded`.

A 60-step smoke run of `timesfm_normalized` through the real pipeline returns
`final_pooled_mse = 289.5`, finite, against 1e12+ before the fix.

**Outcome of the retrain** (job 26669, six variants, ~2.6 h each, all exit 0). The
three normalized variants now agree with each other and with the original variants,
where before they spanned three orders of magnitude:

| variant | pooled MASE before | after |
|---|---|---|
| normalized_natural | 7230.5 | 2.24 |
| normalized_A | 5.81 | 2.40 |
| normalized_B | 1195.7 | 2.08 |
| original_natural | 2.24 | 2.06 |
| original_A | 2.36 | 2.00 |
| original_B | 2.26 | 2.01 |

The spread across variants differing only in scale assignment was the instability;
it is gone. Note the consequence for the study's conclusion: with trainable
normalized variants, TimesFM shows *no* normalized-space advantage (see Results
above), so the fix did not rescue a result, it removed a broken one and
revealed a null.

## Is the MOMENT result just a learning-rate confound? No.

Both MOMENT variants trained with identical `lr = 1e-4` and `grad_clip_norm = 1.0`,
and the paper's real-data scale-swap uses a *learning-rate-adjusted*
original-space variant (`results.tex:181`), an adjustment never applied here. That
looked like a serious confound: original-space MSE gradients scale as `b^p`, so
the two variants might simply be running at wildly different effective step sizes.

Measured on identical batches from a fresh init
(`scripts/_diagnose_gradscale.py`):

| variant | median grad norm | fraction above the 1.0 clip |
|---|---|---|
| `moment_normalized` | 8.83e+01 | 1.00 |
| `moment_original` | 2.92e+10 | 1.00 |

The raw gradient-norm ratio is 3.3e8, but **both variants are clipped on 100% of
steps**. Every update in both variants is therefore rescaled to norm exactly 1.0:
the variants already take steps of identical gradient magnitude, and the runs are
in effect the *unit-norm gradient-matched* condition. A separate
grad-norm-matched variant would reproduce a condition the experiment is already in,
so it was not run.

What clipping does *not* equalize is gradient **direction**. Within a batch,
original-space loss weights high-magnitude datasets far more heavily; rescaling
the total norm afterwards leaves that within-batch weighting untouched. So the
42/42 sweep is a directional effect, not a step-size effect.

This matches the synthetic study's own control, which found the same thing
(`results.tex:55`): "Unit-norm gradient rescaling preserves learning dynamics in
a similar fashion to vanilla original-space training, suggesting that the effect
is not solely an artifact of the learning rate or global update magnitude."

A related point worth stating explicitly, because it is the reason no LR sweep
is planned for the natural-mixture setting: in a controlled experiment `b` is
imposed, so the correction `b^p` is known. In natural pretraining each dataset
has its own `b_d` (and under nonstationarity its own `b_{d,t}`), so the required
correction is not a scalar at all. A single global learning rate can absorb one
global factor; it is structurally incapable of undoing per-source heterogeneous
rescaling. That is the claim, and it is why tuning is not a fix.

## Two evaluation defects found in review

Both affect `train.py`'s in-training eval only (the offline rescoring already
avoided them), and both are fixed for the refreshed runs:

1. **Eval batches were built with `scale_assignment`**, so controlled A/B variants
   evaluated on differently-scaled data. The protocol is that `b` is a
   training-time intervention and all variants are scored on identical
   natural-scale windows. Eval batches now pass no scale assignment;
   `scale_assignment` survives only in the two training-batch constructions.
2. **The pooled eval ran all `eval_batches * batch_size` (25,600) windows in a
   single forward.** It happened to fit on an H200 but is fragile. Eval now
   runs through `eval_scale_free`, which keeps the batch on CPU and moves it in
   `EVAL_CHUNK`-sized pieces, and which also toggles `model.eval()` so dropout
   is not sampled at eval time.

## Verification

- `tests/test_losses.py` gained three tests: MASE invariance under a 10x scale
  multiplier (the property that makes it fair across assignments), the
  unusable-window guards, and NaN-tolerant aggregation.
- MASE invariance holds to ~1e-8 (float32 round-off).

## Robust controlled-scale redesign after the v2 audit

The final-step `timesfm_normalized` A/B test was not a valid scale-effect
test. It compared one trained A model with one trained B model, then treated
42 correlated dataset errors from those shared models as independent
replications. The pooled A-minus-B difference also changed sign across the 15
saved checkpoints. The final 38/42 direction therefore reflects a broad
run-level shift at one checkpoint, not replicated scale evidence.

The controlled pipeline now enforces the following contract.

1. Controlled windows are context-standardized before multiplication by
   `b`. This removes raw levels near `1e17` from the intervention and avoids
   float32 subtraction and overflow artifacts.
2. A and B assign one scale to each complete dataset and swap it in the
   complement. Dataset groups are stable and differ in size by at most one.
3. TimesFM divides its selected sigma by the known `b` before deciding whether
   a window is degenerate. Eligibility is therefore fixed before treatment.
4. TimesFM supports first-patch and causal whole-context normalization. The
   latter directly tests the target-relative-to-statistics hypothesis.
5. The primary mechanistic objective is MSE only. The mixed MSE and pinball
   objective remains available as a secondary ecological condition because
   its two components have different scale degrees.
6. Replicated reporting averages dataset effects inside each trained seed and
   computes uncertainty across seeds. It reports final MASE and the area under
   the MASE training curve. It also reports the original-minus-normalized
   difference-in-differences for the scale effect.

The diagnostic campaign uses the 17M configuration, three seeds, both
normalization modes, two loss spaces, and complementary A/B assignments. The
70M confirmation is launched only after the normalized negative controls,
optimizer-step counts, and run completeness pass.

## Replicated 17M diagnostic

Slurm array 27030 completed all 24 cells with exit code 0. Every cell attempted
and applied 2,000 optimizer updates, with zero skipped updates. The estimator
first averages paired per-dataset effects within each seed, then computes a
Student t interval across the three independent seeds. Positive scale effects
mean mean MASE at `b = 1` exceeds mean MASE at `b = 10`.

The raw-MASE estimates are too uncertain for a final claim at this model size.
All four final-step confidence intervals include zero. The diagnostic therefore
served as a pipeline check and motivation for the 70M confirmation rather than
as reportable scale evidence.

This remains a controlled TimesFM-architecture experiment, not an exact
reproduction of native TimesFM pretraining. It supervises the final forecast
patch with MSE only. Native prefix supervision and the mixed point and quantile
objective require separate ecological-validity checks.

The first 70M submission, Slurm array 27065, exposed a late finalization bug.
All eight seed-0 cells finished 30,000 updates and saved their checkpoints, but
then failed because the legacy through-step-2,000 AUC required two evaluations
at or before step 2,000. This schedule evaluated every 2,000 steps and therefore
had only one eligible point. The remaining array was cancelled after seed 0 so
the same known failure would not consume four more seed waves.

The finalization guard now emits that legacy AUC only when two eligible points
exist. More importantly, each evaluation writes `history.json` immediately.
Trajectory data therefore survive any later summary or checkpoint failure. A
regression test forces finalization to raise and confirms that the history file
already exists. The complete suite passes 71 tests.

## Replicated 70M controlled-scale result

Slurm array 27140 completed four seeds and 32 requested conditions. The fifth
seed was cancelled before it started. Every completed condition exited with
code 0, attempted and applied 30,000 updates, skipped zero updates, recorded
15 evaluations, and saved its final checkpoint.

Positive scale effects mean that a dataset trained at `b = 1` has higher MASE
than the same dataset trained at `b = 10`. Intervals use the four independently
trained seed means.

| normalization | condition | final MASE effect | 95% CI half-width | MASE AUC effect | 95% CI half-width |
|---|---|---:|---:|---:|---:|
| first patch | original | 0.3434 | 0.2882 | 0.2339 | 0.0469 |
| first patch | normalized | -0.0727 | 0.4369 | 0.1006 | 0.4197 |
| whole context | original | 0.4439 | 0.2784 | 0.4458 | 0.1709 |
| whole context | normalized | 0.0253 | 0.3006 | -0.0016 | 0.0556 |

The original-space variants show positive effects while both normalized-space
confidence intervals include zero. The whole-context MASE-AUC
difference-in-differences is 0.4474 plus or minus 0.2230. The final-step
difference-in-differences is positive in both modes but remains too uncertain
to exclude zero. Raw MASE therefore supports the scale-weighting mechanism most
clearly over the training trajectory rather than at one final checkpoint.

Predictive performance depends strongly on how the normalization statistics
are chosen.

| normalization | normalized minus original final MASE | 95% CI half-width | MASE AUC contrast | 95% CI half-width |
|---|---:|---:|---:|---:|
| first patch | 1.8061 | 1.0364 | 2.2181 | 0.4816 |
| whole context | -0.2345 | 0.3356 | -0.2054 | 0.0701 |

First-patch normalized loss is clearly worse in both final MASE and MASE AUC.
Whole-context normalized loss improves MASE AUC, while its final-step contrast
remains uncertain. Whole-context statistics therefore remove the first-patch
target-coordinate failure over the training trajectory.

The total cross-dataset inequality result is not a confirmation. First-patch
normalized loss lowers MASE Gini by 0.0867 plus or minus 0.0610, but raises
MASE IQR by 0.7518 plus or minus 0.2318. It shrinks the extreme tail while
spreading the middle of the distribution, and its forecast error is much
worse. Whole-context normalized loss changes Gini by -0.0255 plus or minus
0.0693 and MASE IQR by -0.0070 plus or minus 0.0384. Both whole-context
inequality measures are consistent with no change.

The primary causal result should therefore be the paired scale effect and its
difference-in-differences. Raw cross-dataset Gini also contains genuine
differences in dataset difficulty and does not isolate scale-induced disparity.

The matched natural-training extension is Slurm array 27223. It uses no
artificial scale assignment. Its original-space variants skipped between
15,246 and 16,922 optimizer updates in the completed seeds, while normalized
variants skipped none. The conditions therefore did not receive comparable
training and their forecast errors are not reportable. The remaining seed was
still incomplete when this note was updated.

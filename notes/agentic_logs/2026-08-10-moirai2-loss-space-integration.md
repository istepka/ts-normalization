# Moirai 2.0 loss-space integration

## Purpose

Moirai 2.0 adds a fourth forecasting architecture to the loss-space study,
alongside MOMENT, TimesFM, and Chronos-2. It tests whether the loss-space
result generalizes to a decoder-only, causally-attended, patch-token
architecture with a native multi-token prediction head, distinct from all
three existing arms.

This is a pretraining experiment from random initialization. It does not use
the published Moirai-2.0-R-small weights and it does not claim to reproduce
the published Moirai 2.0 training run.

## Model

The implementation vendors the relevant subset of `uni2ts==2.0.0`
(Salesforce's Moirai package) into
`src/tsfm_pretraining/vendor/moirai2/` rather than depending on the PyPI
package directly: `uni2ts`'s published pins (`torch<2.5`, `numpy~=1.26.0`,
`datasets~=2.17.1`) conflict with this project's dependencies, the same
situation MOMENT and TimesFM-v1 were already vendored for. See
`src/tsfm_pretraining/vendor/moirai2/REVISION` for the exact file list and
deviations (jaxtyping annotations stripped, relative imports, a few unused
classes dropped).

The configured model uses the official Moirai-2.0-R-small dimensions
(d_model=384, d_ff=1024, 6 layers, patch_size=16, RMSNorm, rotary time
position encoding, binary variate attention bias, GLU feed-forward), about
5M parameters. Instance normalization (`PackedStdScaler`) computes loc/scale
from the context only, the same shape of intervention as Chronos-2's
`instance_norm` and MOMENT's RevIN.

Official sources are listed below.

- https://github.com/SalesforceAIResearch/uni2ts
- https://huggingface.co/Salesforce/moirai-2.0-R-small

## Window length and prediction horizon

Moirai 2.0 natively predicts only `num_predict_token` patches (4 patches ×
16 = 64 steps for the official small config) from the last context token in
a single forward pass; longer horizons require an autoregressive rollout
that feeds the model's own predictions back in as additional context (see
`uni2ts/model/moirai2/forecast.py`, not vendored here since training only
needs the direct single-step forward).

This project's canonical GiftEvalPretrain window index (512 context, 128
prediction) is reused unchanged, so the dataset/window corpus stays
identical across all four TSFM arms (about 55 of 152 datasets survive the
640-step window-length filter). Moirai 2.0 trains and evaluates on only the
first `moirai2.predict_horizon` (64) steps of each window's 128-step target,
using the model's native single-shot forward pass (no autoregressive
rollout), rather than shrinking the shared window index or extending the
training loop with teacher-forced unrolling. The paper's per-model
normalization/MASE/Gini comparisons do not require every arm's eval horizon
to match; only the dataset/window corpus needs to.

## Loss comparison

`moirai2_normalized` is the native Moirai 2.0 objective: quantile (pinball)
loss on the model's own instance-normalized predictions and target.

`moirai2_original` uses the same normalized predictions from the same
forward pass, inverted with the same loc/scale statistics the model computed
internally, and computes the same quantile loss against the target in its
original units.

Nothing else changes between the two conditions. They share initialization,
window order, optimizer, learning rate, training length, and evaluation
data.

## Evaluation

The model uses the existing GiftEvalPretrain window index and natural-scale
evaluation. It reports per-window nMSE and MASE (computed over the 64-step
horizon, not the full 128-step window) and their per-dataset values and Gini
coefficients at every evaluation checkpoint, matching MOMENT/TimesFM/
Chronos-2's reporting.

## Replication

The launcher (`scripts/run_moirai2_pretraining.sbatch`,
`scripts/submit_moirai2_pretraining.sh`) mirrors Chronos-2's: four seeds by
default (0-3), six conditions per seed (native/original × natural-mixture,
plus native/original × controlled-scale A/B), Slurm array capped at eight
concurrent GPU jobs. Not submitted during implementation.

## Verification

`tests/test_moirai2_adapter.py` checks the following properties.

- `run_model`'s extraction/reshape of the raw model output at the last
  context-token position matches a manual reimplementation of the same
  indexing directly against the vendored model's own forward pass.
- `moirai2_original`'s point forecast is the exact loc/scale inverse of the
  normalized median-quantile prediction.
- A tenfold controlled-scale change leaves the normalized-condition gradient
  ratio near one and makes the original-condition gradient ratio near ten,
  the same invariant checked for Chronos-2.
- A training step updates the model's output head and reports non-zero loss.

`tests/test_train.py::test_run_moirai2_end_to_end` exercises the full
`train.py` loop (checkpointing, history/summary persistence, controlled-scale
windows) against the synthetic `tiny_corpus` fixture, matching the existing
MOMENT/TimesFM/Chronos-2 end-to-end tests.

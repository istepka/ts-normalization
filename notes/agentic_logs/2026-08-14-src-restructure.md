# src restructured by concern (2026-08-14)

`src/tsfm_pretraining/` had grown into six concerns in one flat package
(models, data, losses, metrics, training, analysis) and `src/loss_space/` held
a parallel, separate copy of the same ideas for the toy and real-data
experiments. Both packages are gone. Their contents now sit under
`src/config`, `src/data`, `src/losses`, `src/metrics`, `src/models`, and
`src/training`, organized by what the code does rather than which experiment
first needed it.

## What moved where

| was | is |
| --- | --- |
| `tsfm_pretraining/{moment,chronos2,moirai2}_adapter.py`, `timesfm_model.py` | `models/{moment,chronos2,moirai2,timesfm}.py` |
| `tsfm_pretraining/vendor/` | `models/vendor/` |
| `loss_space/model.py` | `models/patch_transformer.py` |
| `tsfm_pretraining/{gifteval_corpus,window_index,corpus_audit}.py` | `data/gifteval/{corpus,window_index,audit}.py` |
| `loss_space/data.py` | `data/loss_space.py` + `data/windows.py` |
| `tsfm_pretraining/{train}.py`, `loss_space/train.py` | `training/{tsfm,loss_space}.py` |
| `tsfm_pretraining/scale_free_metrics.py`, `loss_space_aggregation.py`, `scale_free_report.py`, `metric_explorer.py` | `metrics/{scale_free,aggregate,report,explorer}.py` |
| `loss_space/loss.py` | `losses/loss_space.py` |
| `src/configs.py`, `loss_space/configs.py`, `tsfm_pretraining/configs.py` | `config/{base,loss_space,tsfm}.py` |
| `tsfm_pretraining/scripts/` | `src/scripts/` |

`tsfm_pretraining/losses.py` was four unrelated things in one file and split
accordingly: the objectives to `losses/pointwise.py`, the Gini and per-source
breakdowns to `metrics/inequality.py`, log-MSE AUC and steps-to-threshold to
`metrics/convergence.py`, the MASE denominator to `metrics/forecast.py`,
frequency-alias parsing to `data/seasonality.py`, and the gradient-norm and
safe-clipping helpers to `training/gradients.py`. Call sites that used to
write `L.gini_coefficient` now write `inequality.gini_coefficient`, which says
where the definition lives.

`data/loss_space.py` keeps the dataset classes, `Batch`, and
`build_stratified_schedule` (which returns a `Batch`); only the pure array
transforms went to `data/windows.py`, so the dependency runs one way.

## The four quantile losses

One pinball loss existed in four places: `_quantile_loss` byte-identical in the
Chronos-2 and Moirai-2.0 adapters, `_masked_pinball` in TimesFM, and
`losses.pinball_loss` used only by the tests. They are not interchangeable.
`crps_quantile_loss` is twice `pinball_loss` and sums over quantile levels
rather than averaging, a factor of 2Q, and its horizon mean divides by the
horizon length rather than the valid count. Collapsing all four into one
function would have silently rescaled some model's training gradients.

`losses/quantile.py` therefore keeps two functions, one per convention. Four
copies became two definitions with no numerical change. `pinball_loss` grew an
optional `valid` mask so it covers both the masked and unmasked call sites.

Verified by reimplementing all three original functions and checking them
against the shared ones on 200 random trials, including fully masked rows:
bit-identical (`torch.equal`) across every call convention.

Which model should be using which form is a separate question, taken up in
[2026-08-14-tsfm-loss-fidelity.md](2026-08-14-tsfm-loss-fidelity.md).

## Verification

- 87/87 tests pass, the same count as before the restructure.
- `main.py` runs end to end on the synthetic config (trains four setups,
  writes metrics, renders figures).
- All 19 `python -m` targets named by launchers and the README resolve, and
  every sbatch and sh file passes `bash -n`.
- `ruff check` and `ruff format --check` are clean apart from one pre-existing
  `C408` in the MOMENT adapter, untouched here.

## The ruff exclude was pointing at a path that no longer exists

`pyproject.toml` excluded `src/tsfm_pretraining/vendor` from lint and format so
the pinned upstream source cannot drift from its `REVISION` commit. Moving
vendor to `src/models/vendor/` made that exclude stale, and the first format
run reformatted 12 vendored files. Reverted, and the exclude now names
`src/models/vendor`. Worth remembering that this exclude is path-coupled: any
future move of `vendor/` has to update `pyproject.toml` in the same commit.

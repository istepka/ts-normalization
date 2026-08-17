# TSFM evaluation harness (2026-08-16)

Scores trained checkpoints against the six held-out suites fixed in
`notes/agentic_logs/2026-08-14-holdout-m1-m3.md`. Forecasting only, so
TimesFM, Chronos2, and Moirai2. MOMENT is a reconstruction model with no
forecast head and is out of scope.

## Design

The harness never imports `src/data/gifteval/window_index.py`. That module
manufactures training windows by discarding whatever does not fit (stride,
`min_valid_fraction`, the 640-point floor that keeps only 55 of 152
directories). An eval suite is given rather than manufactured, so the
harness has its own loaders and every series a suite defines is scored.

### Eval modes

Three modes, reported separately and never pooled.

| Mode | Protocol | Metrics |
|---|---|---|
| `native` | official per suite, ragged history, official horizon | accuracy |
| `fixed` | 512 context + 128 horizon, long series only | accuracy |
| `rolling` | tail of each series, stride < H | accuracy + stability |

`native` is the headline. `fixed` exists as a cross-check against the
training-time eval and as insurance that a `native` bug is visible as a
divergence rather than as a plausible wrong number. `rolling` exists only
because the stability metrics require overlapping windows.

Accuracy metrics are nMSE, MASE, MAE, CRPS, WQL, MAPE (plus sMAPE).
Stability metrics are Excess Volatility (EV) and symmetric Forecast
Percentage Change (sFPC).

### Why stability needs its own mode

EV and sFPC compare a forecast of date d made at window t against the
forecast of the same date made at t+1, so they need overlapping windows and
raise when `stride >= H`. Every official protocol here is single-window
(M1/M3/M4/Tourism/Favorita forecast the last H points once), and GIFT-Eval
rolls with `distance=prediction_length`, which is exactly non-overlapping.
So EV/sFPC are undefined on all six official protocols and `rolling` is a
harness-defined mode with its own declared stride.

### No rollout is required

Max horizon per suite is 48 (M4 hourly), 18 (Monash), 16 (Favorita), and 60
(GIFT-Eval short, S freq, `Term.SHORT` multiplier 1). All fit inside every
model's native 128-step horizon, so `predict` truncates and never rolls out
autoregressively. This holds only because GIFT-Eval is restricted to the
short term; medium (10x) and long (15x) would need rollout.

### Model abstraction

`src/eval/` holds no model-specific code. `src/eval/protocol.py` defines a
`Forecaster` protocol, and each of `src/models/{timesfm,chronos2,moirai2}.py`
grows a `build_forecaster(cfg, ckpt_path, device)` next to the
`build_*_model` / `make_batch` / `forward` / `training_step_metrics` set
those three files already share. `predict.py` holds only the
`cfg.model -> module` registry plus the batching and left-padding loop.

### Conventions fixed here

- CRPS uses the doubled sum-over-quantiles form (`crps_quantile_loss`),
  not the mean-over-quantiles `pinball_loss`. The two differ by 2Q and
  `src/losses/quantile.py` carries both because each model trains on its
  own. Eval must use one form for all three models or the column is not
  comparable across models.
- Eval metrics are numpy and live in `src/metrics/`. `src/losses/` stays
  torch and untouched, those being training objectives with autograd.
- MAPE is reported with a per-suite coverage fraction (points with nonzero
  actual), because Favorita is 20.5% NaN filled to 0 plus 0.3% literal
  zeros and its MAPE denominator is mostly degenerate. sMAPE is reported
  alongside.

## Todo

- [x] 1. `src/metrics/eval_losses.py`: numpy `quantile_loss` with
      `aggregate=None` element-wise output, plus `weighted_quantile_loss`
      (WQL) in the GluonTS normalization. Pinned to the torch
      `pinball_loss` by test, the two differing by exactly 2Q.
- [x] 2. `src/metrics/stability.py`: `reshape_windows_by_date`,
      `excess_volatility`, `forecast_percentage_change`. See the EV pairing
      correction below.
- [x] 3. `src/metrics/accuracy.py`: nMSE, MASE, MAE, MSE, CRPS, WQL, MAPE,
      sMAPE, all per series over a ragged left-padded batch, plus `pool`.
      nMSE and MASE reuse the training definitions (context variance and
      in-sample seasonal-naive MAE) so `fixed` mode stays comparable to
      training-time eval. Calibration measured: a seasonal-naive forecast
      scores MASE 1.046 mean / 0.988 median, and a one-step-misaligned
      horizon scores 97.5, so the baseline check is a sharp detector of
      horizon misalignment rather than a marginal one.
- [x] 4. `src/eval/suites.py`: six loaders to a uniform `EvalSeries`, plus
      `load_suite` and `src/scripts/verify_eval_suites.py`. All counts
      confirmed against the real sources, see the table below.
- [ ] 5. `src/eval/protocol.py`: `Forecaster` protocol and registry.
- [ ] 6. TimesFM `build_forecaster`. Verify on M3 native, 3003 forecasts.
- [ ] 7. `src/eval/score.py` + seasonal-naive baseline. Verify baseline
      MASE is approximately 1.0, which is the check that catches horizon
      misalignment.
- [ ] 8. Chronos2 and Moirai2 `build_forecaster`. Verify each reproduces
      its training-time eval loss in `fixed` mode.
- [ ] 9. `rolling` mode wiring, EV and sFPC end to end.
- [ ] 10. GIFT-Eval short native path. Verify a published baseline lands
      near its leaderboard number.
- [ ] 11. `src/scripts/run_tsfm_eval.py` + `conf/eval.yaml`.

## Loaded suites

`uv run python -m src.scripts.verify_eval_suites`, all counts matching the
canonical definitions:

| suite | series | subsets | horizons | history median | under 512 |
|---|---|---|---|---|---|
| m1 | 1,001 | 3 | 6/8/18 | 53 | 1,001 (100%) |
| m3 | 3,003 | 4 | 6/8/18 | 51 | 3,003 (100%) |
| tourism | 1,311 | 3 | 4/8/24 | 102 | 1,311 (100%) |
| m4 | 100,000 | 6 | 6..48 | 97 | 95,103 (95.1%) |
| gifteval | 319,209 | 55 | 6..60 | 665 | 102,854 (32.2%) |
| favorita | 83,207 | 1 | 16 | 1,339 | 6,956 (8.4%) |

The last column is the retrospective justification for dropping the
512-context eligibility rule. Under that rule M1, M3, and Tourism would be
**empty**, and M4 would lose 95% of its series. The rule was a training
constraint read into an evaluation definition, and applying it would not
have produced a worse number so much as no number at all.

Maximum horizon anywhere is 60, against a native model horizon of 128, which
is what makes the no-rollout claim hold.

GIFT-Eval counts instances rather than series, being series times rolling
windows. Its `m4_yearly` holds 22,974 against the official 23,000, which is
the documented GIFT-Eval preprocessing drop and the reason the standalone M4
suite reads the official CSVs instead.

### GIFT-Eval short configs

The benchmark's own `SHORT_DATASETS` list, 55 configs, transcribed into
`GIFTEVAL_SHORT_CONFIGS` and re-derived from the checkout's notebooks by
test. Directory scanning is wrong here: the checkout also carries
`synthetic/*`, which is outside the benchmark, and `jena_weather` exists both
as a leaf and as frequency subdirectories.

The split is reimplemented from gift-eval's `data.py` (prediction-length
maps, window count, non-overlapping windows from the end, multivariate
flattening) rather than imported, because importing it would pull in gluonts
and the gift_eval package, neither of which is installed. Exact leaderboard
parity would want their evaluator instead, which is the open question under
item 10.

### Seasonality is per suite, not per frequency

M4 scores on its own competition seasonality (Daily 1, Weekly 1, Hourly 24)
so the standalone suite stays comparable to the M4 literature, while
GIFT-Eval's `m4_daily` uses the gluonts convention (Daily 7). M3 Other
carries neither a frequency nor a start timestamp in its `.tsf` and declares
period 1. So `EvalSeries` carries an explicit `period` and
`accuracy.per_series_metrics` takes periods rather than frequency strings.

## EV window ordering

EV's pairing depends on how the window axis is ordered, and the reference
implementation this was adapted from carries a different convention, so the
orientation has to be derived here rather than copied.

This harness stacks windows **oldest-first**, window t created before t+1.
For a fixed target date d the source window is `t = (d - h) / stride`, so
ascending h walks backwards in creation date. `h[1:]` is the older forecast
and `h[:-1]` the newer one, meaning `before = h[1:]`. A newest-first
pipeline would need the opposite slices, which is presumably why the
reference reads the other way around.

Getting this backwards flips the sign of the accuracy term, turning the
intended `QL(y, older) - QL(y, newer)` improvement into its negative, so EV
rewards churn. Measured on synthetic forecasts whose error varies along the
window axis:

| pairing | converging | thrashing | |
|---|---|---|---|
| newer as `before` | 0.977 | 0.924 | inverted, converging scores worse |
| older as `before` | 0.111 | 0.181 | thrashing scores worse |

The converging forecast has strictly lower mean absolute error (1.38 vs
2.00). Note both arms land near 0.95 under the wrong pairing, the
mis-signed accuracy term dominating, so this fails quietly rather than
loudly. `test_excess_volatility_penalizes_churn_that_buys_no_accuracy`
pins it. sFPC is unaffected, being symmetric in the two forecasts.

Pair validity is additionally derived from the NaN coverage pattern rather
than from a caller mask alone, so partially covered edge dates do not
contribute zero-filled terms when no mask is supplied.

## Open

Existing checkpoints were trained on an index that included Favorita at
0.19% of windows. Whether that warrants a retrain is undecided and does not
block the harness.

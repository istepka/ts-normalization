# Supervised M-Series and Tourism training

The supervised experiment trains NHITS, NBEATS, and PatchTST through
NeuralForecast on pooled M1, M3, M4, and Tourism series within each
frequency. The launcher covers yearly, quarterly, monthly, weekly, daily, and
hourly frequencies. Each run uses the larger official horizon present in that
frequency as the model horizon. The context is
`max(2 * model_horizon, 2 * seasonal_period)`.

Eligible series reserve their final `2H-1` observations for the test region
and the preceding common model horizon for validation. A series also needs one
complete context and one complete training horizon before validation. Shorter
series are excluded identically from standard and causal runs, with their IDs
recorded in `split_meta.json`. The official final H-step forecast is retained
as a headline comparison. The evaluator also scores every possible H-step
rolling origin in the test region and keeps those rows separate by origin and
official horizon.

The pooled frequency horizon and context determine eligibility before results
are separated by benchmark. Neural methods and statistical references use the
same shared selection, so every benchmark row compares the same series.

The experiment has two normalization modes crossed with the two loss-space
conditions. The `standard` mode uses NeuralForecast scaling on each sampled
window. The `causal` mode disables NeuralForecast scaling with
`scaler_type="identity"` and computes one mean and standard deviation from all
observations available through the end of the input context. The forecast
target is excluded from those statistics.

Within either mode, `sit` uses normalized MAE and `revin` inverts the same
statistics before taking original-space MAE. Causal mode uses the existing SIT
and RevIN transformation interfaces in a manual NeuralForecast training loop,
then reports forecasts in original units. The comparison table therefore has
rows for `standard` and `causal`, crossed with columns for `SIT` and `RevIN`
for each of NHITS, NBEATS, and PatchTST.

Both standard-mode loss conditions use NeuralForecast's standard scaler with
no learnable affine scaling parameters. Only the space in which MAE is taken
changes between them.

Standard and causal runs use the same complete-context window population,
series batch size, sampled-window batch size, Adam optimizer, learning-rate
decay schedule, validation cadence, and early-stopping setting. Causal
validation is processed in bounded batches so pooled M4 runs do not place every
series on the GPU simultaneously.

The convergence follow-up uses a 20,000-update budget, validates every 250
updates, and stops after eight checks without improvement. Both training paths
restore the lowest-validation-loss weights before evaluation. The follow-up
outputs are separate under `outputs/supervised_early_stop`.

Launch from the Auton login host with

```sh
sbatch scripts/train_supervised.sbatch model=nhits condition=sit normalization=standard frequency=M seed=0
```

The matrix submission helper is `scripts/submit_supervised.sh`. Outputs and
checkpoints are written to the scratch-backed path configured in
`conf/supervised.yaml`. Standard runs contain a NeuralForecast checkpoint, and
causal runs contain `causal_model.pt`.

Regenerate the extended-draft result tables from the completed metrics with

```sh
uv run python -m src.plotting.scripts.generate_supervised_mseries_tables \
  --metrics-root /zfsauton/scratch/istepka/ts-normalization/outputs/supervised_early_stop \
  --baseline-root /zfsauton/scratch/istepka/ts-normalization/outputs/supervised_baselines \
  --table-dir /zfsauton2/home/istepka/ts-normalization/overleaf/extended_draft/tables
```

The generator requires the complete 72-run grid and fails on missing,
duplicate, non-finite, or non-positive metric values.

The affine-free and shared-eligibility correction was rerun on 2026-08-19.
Training jobs `33591` through `33608`, evaluation jobs `33660` through
`33677`, and reference array `33609` completed successfully. The dated result
snapshot is stored under
`outputs/supervised_result_snapshots/2026-08-19_post_fairness_fix` on scratch.
Both fitted references match the neural population across all 215 grouped
evaluation rows. Each reference fitted all 934,503 forecast cases.

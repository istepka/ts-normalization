# GiftEvalPretrain loss-space experiments with MOMENT and TimesFM

## Summary

Run two staged TSFM pretraining studies on the same local GiftEvalPretrain corpus.

1. Train MOMENT with masked reconstruction.
2. Train a reduced TimesFM-1.0 architecture with forecasting loss.

For each model, compare normalized-space and original-space loss under identical initialization, data order, masking, optimizer, and training budget.

The main claim will be that loss-space effects can be traced during actual TSFM pretraining on a large heterogeneous public corpus. The experiments will not claim to reproduce published MOMENT or TimesFM checkpoints.

GiftEvalPretrain is the shared corpus because it is already available locally, spans many domains and frequencies, and is explicitly intended for non-leaking foundation-model pretraining.

Sources:

- [GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)
- [MOMENT paper](https://arxiv.org/abs/2402.03885)
- [MOMENT repository](https://github.com/moment-timeseries-foundation-model/moment)

## Corpus and data pipeline

### Main corpus

Use the local GiftEvalPretrain copy.

The main experiment will use only univariate series so MOMENT and TimesFM consume the same canonical window population. Multivariate data is excluded from the primary comparison because TimesFM-1.0 is univariate.

Record the following metadata for every canonical series:

- dataset identifier
- domain
- frequency
- series identifier
- start timestamp
- target length
- missing-value mask
- source file checksum
- preprocessing version

Do not concatenate raw points and sample proportionally to storage size.

Build a stratified sampler that balances dataset, domain, frequency, and series identity. The exact sampling weights must be written into the resolved configuration and copied into every run summary.

Log the cumulative number of windows actually processed per dataset, domain, and frequency over training, not only the configured sampling weights. Realized exposure can drift from the configured weights because of filtering, epoch boundaries, and per-worker sampling variance.

### Canonical window index

Create one shared index of eligible univariate windows.

Each record must contain:

- dataset
- domain
- frequency
- series identifier
- window start
- context length
- prediction length
- valid-value mask
- context mean
- context standard deviation
- deterministic masking seed
- deterministic augmentation seed

Both models must consume the same base index. Their model-specific masking and target construction may differ, but must be generated deterministically from the shared record.

Split by series before creating windows. No series may occur in both training and validation.

### Scale intervention

For controlled runs, create complementary scale assignments within every dataset and frequency cell.

- Assignment A uses `b=1` for half of the eligible windows and `b=10` for the other half.
- Assignment B swaps the assignments.
- The underlying normalized shapes, series, windows, and order remain identical.

The scale is applied before the model's normalization path. This preserves normalized inputs while changing the original-space residual.

## Dispersion and equity metrics

Track the Gini coefficient of per-source error alongside every convergence curve, not only at the final checkpoint. Compute it over the same set of sources used for the per-source curves, with dataset as the primary unit. Follow the same convention as the synthetic loss-space Gini table already in the paper: report it through the first 2,000 steps, over all logged checkpoints, and at the final step.

Domain-level and frequency-level Gini are secondary breakdowns. Reuse the same formula over domain or frequency means instead of dataset means.

Record the number of sources `n` feeding each Gini computation in every run summary, since it varies by breakdown.

Track the unweighted mean error across sources separately from the pooled, natural-mixture-weighted global error. Do not assume the two are equal. In the synthetic toy experiment the validation set was balanced per source, which made the pooled global metric numerically identical to the unweighted mean, a fact that only became clear after computing both directly. GiftEvalPretrain datasets vary enormously in size, so keep the natural-mixture validation set in its natural, imbalanced proportion rather than rebalancing it, so the comparison between the pooled global metric and the unweighted per-source mean is informative rather than vacuous. A separate balanced-validation variant can be added later if a clean apples-to-apples comparison to the synthetic toy is needed.

Report the per-dataset windows-processed counts (see Main corpus) alongside Gini at the same checkpoints. Training exposure is a confound for the equity story that is separate from the loss-space mechanism: if the sampler under-serves a source relative to its configured weight, that source can look inequitably fit for reasons that have nothing to do with normalized-space versus original-space loss. A source with persistently high Gini contribution despite on-target exposure is the clean evidence for the loss-space claim; a source with high Gini contribution and under-target exposure is not.

## MOMENT stage

### Implementation

Reuse the official MOMENT PyTorch implementation and research pretraining code where possible. Add a project adapter rather than reimplementing the model.

The adapter must expose:

- deterministic masked-window batches
- normalized targets
- original-space targets
- reconstruction predictions
- per-example losses
- pre-optimizer gradient norms

Pin the MOMENT source revision and record it in every run.

### Objective conditions

Run two conditions.

- `moment_normalized` keeps MOMENT's native normalized reconstruction objective.
- `moment_original` keeps the exact same forward pass and masked reconstruction task, but inverse-transforms predictions and targets before computing MSE.

The second condition is an explicit counterfactual intervention on MOMENT's loss space. It is not presented as MOMENT's published native objective.

The expected controlled-scale gradient ratio is approximately `b^2` for MSE.

### MOMENT metrics

Log by dataset, domain, and frequency:

- reconstruction MSE in normalized space
- reconstruction MSE in original space
- masked and unmasked error
- per-source convergence curves
- log-MSE area under the curve
- steps to fixed error thresholds
- gradient norms before clipping
- gradient norms after clipping
- clipping frequency
- per-source error dispersion, Gini coefficient, and unweighted mean across sources (see Dispersion and equity metrics)

The primary MOMENT figure will compare per-source convergence under normalized and original loss. The controlled scale replay will provide the causal attribution.

## TimesFM stage

### Implementation

Port the released legacy TimesFM-1.0 architecture into the current PyTorch project.

Preserve:

- decoder-only causal structure
- input patch length 32
- output patch length 128
- instance normalization
- patch masking
- residual input and output blocks
- layer normalization
- quantile and point output heads
- native inverse-transform and loss path

Use a reduced configuration for the main run. The 17M configuration is the smoke test. The 70M configuration is the primary run if throughput permits.

Add a numerical parity test against a reference generated from the pinned legacy implementation.

### Objective conditions

Run:

- `timesfm_native_original`
- `timesfm_normalized`

The native condition computes the released MSE and quantile or pinball terms after inverse transformation.

The normalized condition computes the same terms before inverse transformation.

The forward pass, model initialization, data order, optimizer, masking, and training schedule must remain unchanged.

Log MSE and pinball components separately because their expected scale degrees differ.

- MSE contribution scales approximately with `b^2`.
- Pinball contribution scales approximately with `b`.

### TimesFM metrics

Use the same per-source diagnostics as MOMENT, plus:

- MSE gradient norm
- pinball gradient norm
- total gradient norm
- MSE-to-pinball loss ratio
- MSE-to-pinball gradient ratio
- raw-space and normalized-space forecast error

## Execution order

### Phase 1

Audit the local GiftEvalPretrain files.

Produce:

- source inventory
- frequency inventory
- univariate series count
- missing-value statistics
- variance histogram
- source and frequency sampling table

Fail if the local copy cannot be fingerprinted or if the intended univariate sources are missing.

### Phase 2

Run a MOMENT single-GPU smoke test.

Verify batch construction, masking, normalization, reconstruction output shape, loss-space switching, gradient logging, checkpoint loading, and deterministic replay.

### Phase 3

Run the paired MOMENT natural-mixture experiment.

Use the same seeds and shared index for normalized and original conditions.

### Phase 4

Run the paired MOMENT controlled-scale experiment.

Use complementary scale assignments and report the within-cell paired effect.

### Phase 5

Run the TimesFM architecture parity smoke test.

Do not begin TimesFM pretraining until the parity test passes.

### Phase 6

Run the paired TimesFM natural-mixture experiment.

Use the same GiftEvalPretrain source inventory and compatible univariate window index.

### Phase 7

Run the paired TimesFM controlled-scale experiment.

Report separate MSE and pinball scaling.

## Files and interfaces

Add:

- `src/tsfm_pretraining/gifteval_corpus.py`
- `src/tsfm_pretraining/window_index.py`
- `src/tsfm_pretraining/moment_adapter.py`
- `src/tsfm_pretraining/timesfm_model.py`
- `src/tsfm_pretraining/losses.py`
- `src/tsfm_pretraining/train.py`
- `scripts/audit_gifteval_pretrain.py`
- `scripts/build_gifteval_window_index.py`
- `scripts/run_moment_pretraining.sbatch`
- `scripts/run_timesfm_pretraining.sbatch`
- `scripts/aggregate_tsfm_loss_space.py`

Write resolved configurations, source checksums, model revisions, manifest checksums, and run summaries into each output directory.

Do not modify the existing PatchTST experiments.

## Tests and acceptance criteria

### Data tests

- Source and frequency inventories are reproducible.
- Series-level train and validation splits are disjoint.
- Paired conditions consume identical base windows.
- Scale assignments are exact complements.
- Replaying a seed reproduces identical batches.

### MOMENT tests

- Masked reconstruction shapes are correct.
- Native normalized loss matches the official implementation.
- Original-space loss is the only changed computation.
- MSE gradient ratio follows the assigned scale in the controlled test.

### TimesFM tests

- Patching and masking match the legacy reference.
- Inverse transformation is affine-correct.
- MSE and pinball losses are separated correctly.
- Original-space MSE and pinball gradients show their expected scale factors.

### Scientific acceptance

The study is successful if:

- both models train without NaNs
- both paired conditions have complete per-source logs
- original-space controlled runs show scale-dependent gradients
- normalized-space controlled runs remove that direct dependence
- MOMENT and TimesFM results are reported with the same source and window provenance
- conclusions distinguish natural-mixture evidence from controlled causal evidence
- the natural-mixture validation set is imbalanced enough across datasets that the pooled global metric and the unweighted per-source mean can diverge, so their comparison is informative rather than trivially identical
- normalized-space runs show lower per-source Gini than original-space runs at matched checkpoints, or the natural-mixture results explain why not

## Optional Time Series Pile check

Do not use Time Series Pile for the primary experiment.

If the GiftEvalPretrain results are successful, run a small MOMENT-only replication on a documented subset of the public Time Series Pile. This check is useful for the narrower statement that the effect also appears on MOMENT's own pretraining corpus.

It is not required for the main paper result.

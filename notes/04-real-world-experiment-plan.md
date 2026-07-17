# 04 — Real-world experiment plan

Goal: bridge the clean synthetic loss-space result to real time-series data without
immediately jumping into a fully confounded multi-dataset pretraining setup.

## Proposed ladder

1. **Real-shape controlled-scale.** Load windows from real series, normalize each base
   window with its context statistics, then create category copies with the same scales as
   the synthetic experiment (`var1`, `var10`, `var100`). This keeps the normalized real
   shape identical across categories while making amplitude/variance the intended
   intervention.
2. **Variance-binned real windows.** Pool natural real windows and bin them by context
   variance. This is less controlled but tests whether naturally high-variance windows
   dominate original-space training.
3. **Multi-dataset real pretraining toy.** Treat datasets as sources and stratify batches
   across them. This is closest to TSFM pretraining but introduces source difficulty as a
   confounder, so it should come after the cleaner checks.
4. **Correlation analysis.** For each source/bin, scatter context variance against init
   gradient magnitude, convergence speed, steps-to-threshold, and final nMSE. Compare
   original-space and normalized-space loss with Spearman correlation.
5. **Pretrained TSFM normalization override.** Use Chronos/TimesFM-style zero-shot
   evaluation to show practical relevance, but treat it as downstream support rather than
   direct evidence of training loss-space bias.

## Implemented first: real-shape controlled-scale

Config switch: `data.kind=real_shape_scaled`.

Input format: an `.npz` file containing the key named by `data.real_shape_key` (default:
`windows`). The array can be:

- `[N, context+horizon]`: precomputed windows.
- `[N, T]` or `[T]`: raw series, converted to sliding windows.

For every loaded base window, the dataset computes context mean/std, normalizes the full
context+horizon window by those context statistics, and then creates the configured category
copies:

```text
scaled_window_c = data.mean + category.scale * normalized_real_shape
```

The `original_equalvar` control still sets every category scale to `1.0`, and the batch
schedule remains stratified across categories. Validation windows are a fixed random split
from the same base-window pool using `VAL_SEED`.

Expected readout:

- `normalized`: categories should learn similarly because normalized shapes are identical.
- `original`: if the loss-space mechanism survives real shapes, init gradient magnitude and
  early convergence should follow the scale/variance ordering.
- `original_equalvar`: category spread should collapse.
- `original_gradmatch`: any remaining spread is not just global gradient magnitude.

Concrete local run using the SSTC electricity windows:

```sh
sbatch --job-name=loss_space_realshape scripts/run.sbatch \
  wandb.experiment=real_shape_scaled_electricity_30k_sgd \
  data.kind=real_shape_scaled \
  data.real_shape_path=/zfsauton2/home/istepka/sstc/experiments/h1/data/128_electricity.npz \
  data.real_shape_key=data
```

The local SSTC arrays use key `data`. Some real datasets contain flat windows; the loader
filters non-finite or zero-context-std base windows before normalization and fails if none
remain.

## Implemented second: variance-binned real windows

Config switch: `data.kind=real_variance_binned`.

This mode loads natural real windows from the same `.npz` format, filters non-finite and
zero-context-std windows, splits train/validation with `VAL_SEED`, then bins windows by
training-set context standard deviation quantiles. With three configured categories, the
bins are named `low_var`, `mid_var`, and `high_var`.

Unlike the real-shape controlled-scale setup, this does **not** copy the same shape across
scales. It keeps natural real windows in their natural scale bins, so source difficulty and
variance are mixed. This is the realistic-but-noisier check.

The `original_equalvar` control keeps the same bin membership but context-normalizes every
window to unit scale. If the original-space spread collapses in that control, the spread is
more plausibly about scale rather than bin-specific shape difficulty.

Concrete local run using the SSTC electricity windows:

```sh
sbatch --job-name=loss_space_varbins scripts/run.sbatch \
  wandb.experiment=real_variance_bins_electricity_30k_sgd \
  data.kind=real_variance_binned \
  data.real_shape_path=/zfsauton2/home/istepka/sstc/experiments/h1/data/128_electricity.npz \
  data.real_shape_key=data \
  data.real_value_scale=0.0001
```

The raw electricity windows have a very wide context-std range (roughly `0.5` to `6.4e4`).
At `real_value_scale=1.0`, original-space SGD diverges immediately; `0.0001` preserves the
relative variance-bin ordering while keeping the absolute original-space loss scale finite
in a 100-step smoke test.

## Eight-dataset variance-bin robustness rerun

The Electricity-only run used a random split after creating overlapping windows. For raw
`[N, T]` arrays, the corrected loader now splits source-series rows first and only then
extracts sliding windows. This prevents train and validation windows from sharing nearly
all of their observations.

The robustness rerun repeats the variance-bin experiment independently on eight datasets:
Electricity, Traffic, Solar 10 Minutes, Taxi 30 Minutes, Wind Farms, Pedestrian Counts,
KDD Cup 2018, and FRED-MD. Quantile bins are computed separately from each dataset's
training windows. The training-only p99 context standard deviation is scaled to
`0.6475868966` for every dataset, matching the stable Electricity configuration while
preserving each dataset's internal low/mid/high variance ratios.

Seeds are averaged within each dataset first. Plots and endpoint summaries then report the
mean and 95% Student-t confidence interval across the eight dataset-level results, so large
datasets do not receive more inferential weight simply because they contain more windows.
This tests whether the Electricity finding is robust across domains; it does not by itself
remove the variance--forecast-difficulty confound.

The single job `scripts/run_variance_bins_8datasets.sbatch` runs all eight datasets
sequentially and invokes `scripts/aggregate_variance_bins.py` to produce the
cross-dataset plots and numerical summary.

## Eight-dataset scale-swap crossover

The next experiment removes the natural variance--difficulty confound by making scale a
controlled intervention. Each of the same eight datasets is split with the
leakage-free series-level split, context-normalized window by window, and sampled
equally within every batch. Assignment A gives scale `b=1` to Electricity, Traffic,
Solar 10 Minutes, and Taxi, and `b=10` to Wind Farms, Pedestrian Counts, KDD Cup 2018,
and FRED-MD. Assignment B swaps every scale while retaining dataset order, seeded
sample schedules, and initialization.

Only the normalized-space negative control and original-space treatment are run. The
primary statistic is the paired within-dataset difference in mean log10-nMSE AUC through
step 2,000:

```text
AUC(b=1) - AUC(b=10)
```

A positive value means the high-scale assignment converged faster. Original-space loss
should produce a positive paired effect and an initialization gradient ratio near
`10^2=100`; normalized-space loss should remain invariant to the assigned scale. The 95%
Student-t interval uses the eight paired dataset effects as independent units after
averaging the five seeds within each dataset.

The job `scripts/run_scale_swap_8datasets.sbatch` runs both assignments and then invokes
`scripts/aggregate_scale_swap.py`. It produces early convergence curves with
dataset-level confidence intervals, a paired AUC figure, an
initialization-gradient-ratio figure, and a JSON numerical summary.

# Leakage-free real-window split and eight-dataset rerun

## Problem

The real-data loader created overlapping sliding windows before randomly splitting train
and validation. Adjacent 80-step windows from the same source series could therefore share
79 observations across the split, making validation curves optimistic.

## Correction

- Raw `[N, T]` arrays now split source-series rows before window extraction.
- Precomputed `[N, 80]` arrays split independent window rows.
- Single raw series use non-overlapping contiguous train/validation time segments and fail
  if either segment cannot produce an 80-step window.
- Filtering, variance thresholds, and scale statistics use the resulting training split.

## Rerun

The immediate rerun repeats the variance-bin experiment independently across eight real
datasets. Each dataset retains its own low/mid/high quantile bins. Five seeds are averaged
within dataset; reported means and 95% confidence intervals use the eight datasets as the
independent units.

Per-dataset value multipliers use training-only context statistics to map p99 context
standard deviation to the stable Electricity value `0.6475868966`. The multipliers and
dataset paths are fixed in `scripts/run_variance_bins_8datasets.sbatch`.

The single Slurm job runs all datasets sequentially, then invokes the aggregator itself.
The job ID and output path will be added after submission.

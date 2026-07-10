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

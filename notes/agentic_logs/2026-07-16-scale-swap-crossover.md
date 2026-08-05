# Eight-dataset scale-swap crossover

Implemented `data.kind=real_scale_swap`, which loads the eight configured real datasets
through the leakage-free splitter, context-normalizes each window, and applies one
controlled scale per dataset. Equal source sampling is enforced by the existing
stratified batch scheduler with batch size 32.

Added a core-only setup selection so this experiment runs only normalized-space and
original-space loss. The two assignments use the same dataset ordering and random seeds,
which makes sampled windows and model initialization identical before the scale swap.

Added `scripts/run_scale_swap_8datasets.sbatch` and
`scripts/aggregate_scale_swap.py`. The aggregation pairs each dataset with itself across
assignments, reports early log10-nMSE AUC through step 2,000, and creates three figures:
paired convergence curves, paired AUC effects, and initial gradient ratios.

The first full run, job 18898, completed in about 26 minutes but assignment A's
original-space setup diverged for seeds 0, 2, and 4. The scale-swap-only config now uses a
larger model (`d_model=128`, four layers, eight heads, feedforward width 256) and lowers
the shared learning rate to `1e-4`. Training and aggregation now fail on non-finite
metrics instead of writing a `NaN` summary.

Before another full run, `scripts/run_scale_swap_smoke.sbatch` exercises all five seeds,
both loss modes, and both assignments through step 200 with W&B disabled. This exceeds
the latest first-failure step from job 18898 (step 170).

Job 18948 completed both stabilized assignments. The reanalysis now treats the first
2,000 steps as a zoom rather than the sole view: it includes full 30,000-step convergence
panels, an early companion, eight original-loss dataset panels, the paired log-nMSE
difference over the complete trajectory, and a simplified gradient-ratio bar chart. The
JSON summary retains early AUC and adds full-horizon AUC and the final paired effect.

The LR-adjusted follow-up adds `original_lr_adjusted` with learning rate
`1e-4 / 50.5 = 1.9801980198e-6`. This matches the average step scale implied by four
`b=1` and four `b=10` sources while preserving their 100:1 relative loss weighting.
`scripts/run_scale_swap_lr_adjusted.sbatch` runs only this missing A/B setup and merges
it with job 18948 for three-way plots.

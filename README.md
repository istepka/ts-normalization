# TSFM-normalization project

## Configs

The project uses Hydra configs in `conf/`. YAML remains the record of each
experiment, while `src/configs.py` supplies structured schemas that reject
unknown keys, wrong types, and missing required values at runtime.

There are two config families.

- `config.yaml` is the synthetic loss-space experiment.
- `scale_swap.yaml` inherits the synthetic config and overrides it for the
  real-data scale-swap experiment.
- `tsfm_chronos2.yaml`, `tsfm_moirai2.yaml`, `tsfm_moment.yaml`, and
  `tsfm_timesfm.yaml` define the four TSFM runs.
- The TSFM configs inherit shared corpus, window-index, training, and W&B
  defaults from `tsfm_base.yaml`. Their remaining fields are model-specific
  architecture and experiment choices.

Run a config by selecting its name and adding Hydra overrides as needed.

```sh
uv run python main.py --config-name=scale_swap

uv run python -m src.tsfm_pretraining.train \
    --config-name=tsfm_timesfm \
    device=cuda \
    timesfm.config_size=70m \
    train.steps=30000
```

TSFM runs save the resolved configuration as `resolved_config.yaml` in their
output directory. This is the configuration to use when checking exactly what
an experiment ran with.

## Scripts

#### `build_gifteval_window_index.py` / `.sbatch`

```sh
uv run python -m scripts.build_gifteval_window_index \
    --corpus-root /path/to/giftevalpretrain_full \
    --output outputs/gifteval_window_index/context512_pred128.parquet \
    --context-length 512 --prediction-length 128 --stride 512
```

Pre-builds and caches the canonical GiftEvalPretrain window index as a parquet file. Every paired training run (normalized vs. original, MOMENT vs. TimesFM, scale A vs. B) reads this same cache instead of independently re-scanning the corpus, so paired conditions are guaranteed to train on identical windows.

#### `audit_gifteval_pretrain.py`

```sh
uv run python scripts/audit_gifteval_pretrain.py \
    --corpus-root /path/to/giftevalpretrain_full \
    --output-dir outputs/gifteval_audit
```

Audits the local GiftEvalPretrain corpus and fails loudly if it can't be fingerprinted or the domain mapping is incomplete. Produces a per-dataset inventory, frequency and missing-value breakdowns, a variance histogram, and the sampling table used to configure `train.py`'s dataset weights.

#### `run_pretraining.sbatch`

```sh
MODEL=moment sbatch scripts/run_pretraining.sbatch   # or timesfm | chronos2 | moirai2
```

The shared Slurm GPU launcher for MOMENT, TimesFM, Chronos-2, and Moirai-2.0 loss-space pretraining, selected by `MODEL`. Each array task trains one (condition, experiment_kind[, scale_assignment]) run in parallel; prefer submitting it via `submit_pretraining.sh` rather than directly.

#### `submit_pretraining.sh`

```sh
MODEL=moment scripts/submit_pretraining.sh
```

Submits the full dependency chain for a model: build the window index if missing, submit `run_pretraining.sbatch`, then (for MOMENT/TimesFM) submit `aggregate_pretraining.sbatch` once the array job completes.

#### `aggregate_pretraining.sbatch`

```sh
MODEL=moment JOBTAG=gifteval_moment_12345 sbatch scripts/aggregate_pretraining.sbatch
```

Aggregates the 6 runs produced by one `run_pretraining.sbatch` array job (MOMENT or TimesFM) into comparison tables, by calling `aggregate_tsfm_loss_space.py` once for the natural-mixture runs and once per controlled-scale condition. Submitted automatically by `submit_pretraining.sh`.

#### `aggregate_tsfm_loss_space.py`

```sh
uv run python -m scripts.aggregate_tsfm_loss_space \
    --run moment_normalized=outputs/.../moment_normalized \
    --run moment_original=outputs/.../moment_original \
    --output-dir outputs/.../aggregate
```

Reads a set of run output directories and writes `comparison.csv`/`.json`: one row per run with final pooled MSE, log-MSE AUC, and per-dataset/domain/frequency Gini. With `--scale-pair`, also writes the paired per-dataset AUC(A) - AUC(B) effect and its 95% confidence interval.

#### `recompute_tsfm_scale_free_metrics.py`

```sh
uv run python -m scripts.recompute_tsfm_scale_free_metrics \
    --run moment_original_A=outputs/gifteval_moment_..._A \
    --run moment_original_B=outputs/gifteval_moment_..._B \
    --output-dir outputs/scale_free_metrics
```

Rereads saved checkpoints and rescores every run on identical, scale-free nMSE/MASE definitions, since the original training-time eval used each variant's own loss space and so wasn't comparable across normalized vs. original runs. Writes `scale_free_metrics.csv` (per-checkpoint) and `final_per_dataset.json` (final-step per-dataset breakdown).

#### `report_scale_free_tables.py`

```sh
uv run python -m scripts.report_scale_free_tables \
    --metrics-dir outputs/tsfm_scale_free_metrics
```

Consumes the output of `recompute_tsfm_scale_free_metrics.py` and builds the final result tables: a final-step summary, paired scale-A-vs-B effects, and normalized-vs-original head-to-head comparisons. Writes `summary.csv`, `paired_effects.json`, and `head_to_head.json` into the same metrics directory.

#### `build_metric_explorer.py`

```sh
uv run python -m scripts.build_metric_explorer \
    --metrics-dir outputs/tsfm_scale_free_metrics_v2 \
    --template scripts/metric_explorer_template.html --out explorer.html
```

Inlines a recomputed metrics directory's data into the explorer template, producing a single self-contained HTML page (no external requests, so it also works published as an Artifact). Lets you interactively browse per-checkpoint and per-dataset metrics across every run variant.

#### `adhoc/`

One-off scripts pinned to specific past run IDs (seed reruns, metric recomputes) that aren't meant to be reused with different arguments. Kept for the record rather than deleted.

#### `reproducibility/`

Scripts backing specific paper experiments, organized by experiment line (e.g. `real_scale_swap/`, `synthetic_loss_space/`).

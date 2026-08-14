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

## Code layout

Python lives under `src/`, organized by what it does. `scripts/` holds only
Slurm launchers and shell wrappers, so nothing in it needs importing.

```text
src/
  configs.py                 structured Hydra schemas
  loss_space/                synthetic loss-space toy
  tsfm_pretraining/          TSFM pretraining, eval, and aggregation
    scripts/                 CLIs for the modules above
  plotting/
    core/                    palette, figure helpers, run registries
    scripts/                 figure and table entry points
  scripts/                   repo-level tooling invoked by launchers
scripts/                     sbatch files and submit wrappers
```

Everything under `src/` runs as a module, for example

```sh
uv run python -m src.plotting.scripts.plot_tsfm_natural_convergence
```

### `src/plotting/`

Everything that draws a figure or builds a paper table.

`core/` holds the shared machinery. `color_palette.json` and `palette.py` carry
the LTS palette, which is installed as matplotlib's default color cycle on
import, so figures share one look without per-script setup. `figures.py` has
`mean_ci`, `save_figure`, the shared axis and legend styling, and
`PAPER_RCPARAMS`, the submission defaults applied on import: TrueType (Type 42)
font embedding rather than the Type 3 fonts matplotlib emits by default and most
venues reject, 300 dpi rasters, opaque backgrounds, and print-safe line widths.
`save_figure` writes a .pdf for the paper and a .png for previews, crops to
`bbox_inches="tight"` with a small pad, and omits the PDF creation date so
re-running a script on unchanged data leaves the file byte identical. Panels
meant to be typeset side by side pass `bbox_inches=None` so their axes stay
aligned. `tsfm_runs.py`
has the four-model run registry and the history readers. `loss_space.py` has the
synthetic loss-space toy figures.

`scripts/` holds the entry points

```sh
uv run python -m src.plotting.scripts.plot_tsfm_natural_convergence
uv run python -m src.plotting.scripts.plot_tsfm_convergence_subfigures
uv run python -m src.plotting.scripts.plot_tsfm_inequality_convergence
uv run python -m src.plotting.scripts.summarize_tsfm_paper_results
```

with per-experiment figure builders under
`src/plotting/scripts/reproducibility/`, organized by experiment line
(`real_scale_swap/`, `real_variance_bins/`, `synthetic_loss_space/`).

### `src/tsfm_pretraining/scripts/`

Thin CLIs over the `src.tsfm_pretraining` modules. Each one's docstring points
at the module that does the work.

#### `build_gifteval_window_index.py`

```sh
uv run python -m src.tsfm_pretraining.scripts.build_gifteval_window_index \
    --corpus-root /path/to/giftevalpretrain_full \
    --output outputs/gifteval_window_index/context512_pred128.parquet \
    --context-length 512 --prediction-length 128 --stride 512
```

Pre-builds and caches the canonical GiftEvalPretrain window index as a parquet file. Every paired training run (normalized vs. original, MOMENT vs. TimesFM, scale A vs. B) reads this same cache instead of independently re-scanning the corpus, so paired conditions are guaranteed to train on identical windows. Its launcher is `scripts/build_gifteval_window_index.sbatch`.

#### `audit_gifteval_pretrain.py`

```sh
uv run python -m src.tsfm_pretraining.scripts.audit_gifteval_pretrain \
    --corpus-root /path/to/giftevalpretrain_full \
    --output-dir outputs/YYYY-MM-DD/analysis/tsfm_pretraining/gifteval_audit
```

Audits the local GiftEvalPretrain corpus and fails loudly if it can't be fingerprinted or the domain mapping is incomplete. Produces a per-dataset inventory, frequency and missing-value breakdowns, a variance histogram, and the sampling table used to configure `train.py`'s dataset weights.

#### `aggregate_tsfm_loss_space.py`

```sh
uv run python -m src.tsfm_pretraining.scripts.aggregate_tsfm_loss_space \
    --run moment_normalized=outputs/YYYY-MM-DD/experiments/tsfm_pretraining/.../moment_normalized \
    --run moment_original=outputs/YYYY-MM-DD/experiments/tsfm_pretraining/.../moment_original \
    --output-dir outputs/YYYY-MM-DD/analysis/tsfm_pretraining/.../aggregate
```

Reads a set of run output directories and writes `comparison.csv`/`.json`: one row per run with final pooled MSE, log-MSE AUC, and per-dataset/domain/frequency Gini. With `--scale-pair`, also writes the paired per-dataset AUC(A) - AUC(B) effect and its 95% confidence interval.

#### `recompute_tsfm_scale_free_metrics.py`

```sh
uv run python -m src.tsfm_pretraining.scripts.recompute_tsfm_scale_free_metrics \
    --run moment_original_A=outputs/YYYY-MM-DD/experiments/tsfm_pretraining/.../moment_original_A \
    --run moment_original_B=outputs/YYYY-MM-DD/experiments/tsfm_pretraining/.../moment_original_B \
    --output-dir outputs/YYYY-MM-DD/analysis/tsfm_pretraining/scale_free_metrics
```

Rereads saved checkpoints and rescores every run on identical, scale-free nMSE/MASE definitions, since the original training-time eval used each variant's own loss space and so wasn't comparable across normalized vs. original runs. Writes `scale_free_metrics.csv` (per-checkpoint) and `final_per_dataset.json` (final-step per-dataset breakdown).

#### `report_scale_free_tables.py`

```sh
uv run python -m src.tsfm_pretraining.scripts.report_scale_free_tables \
    --metrics-dir outputs/YYYY-MM-DD/analysis/tsfm_pretraining/scale_free_metrics
```

Consumes the output of `recompute_tsfm_scale_free_metrics.py` and builds the final result tables: a final-step summary, paired scale-A-vs-B effects, and normalized-vs-original head-to-head comparisons. Writes `summary.csv`, `paired_effects.json`, and `head_to_head.json` into the same metrics directory.

#### `build_metric_explorer.py`

```sh
uv run python -m src.tsfm_pretraining.scripts.build_metric_explorer \
    --metrics-dir outputs/YYYY-MM-DD/analysis/tsfm_pretraining/scale_free_metrics \
    --template src/tsfm_pretraining/metric_explorer_template.html \
    --out explorer.html
```

Inlines a recomputed metrics directory's data into the explorer template, producing a single self-contained HTML page (no external requests, so it also works published as an Artifact). Lets you interactively browse per-checkpoint and per-dataset metrics across every run variant.

### `src/scripts/`

Repo-level tooling that the launchers call. `organize_outputs.py` migrates
outputs into the dated layout described below. `permutation_schedule.py`
generates the balanced complementary assignments for the scale-swap permutation
campaign and is invoked by the `train_permutation_pair_*.sbatch` launchers.

## Slurm launchers

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

#### `adhoc/`

One-off launchers pinned to specific past run IDs (seed reruns, metric recomputes) that aren't meant to be reused with different arguments. Kept for the record rather than deleted.

#### `reproducibility/`

Launchers backing specific paper experiments, organized by experiment line (e.g. `real_scale_swap/`, `synthetic_loss_space/`). Their figure-building counterparts live under `src/plotting/scripts/reproducibility/`.

## Output layout

New outputs use the following path shape

```text
outputs/YYYY-MM-DD/<category>/<experiment>/<run>/
```

The categories are `experiments`, `analysis`, `visualizations`, `data`, and
`diagnostics`. Training artifacts belong under `experiments`. Derived tables
belong under `analysis`. Figures and replots belong under `visualizations` and
keep the source experiment or run name in their path. The shared launchers use
`scripts/output_paths.sh` and pass Hydra's run directory explicitly so Hydra
metadata stays beside the artifacts it describes.

Two paths deliberately sit outside the dated layout. The GiftEvalPretrain
window index is a shared cache every paired run must read from, so it stays at
`outputs/gifteval_window_index/` (override with `INDEX_DIR` or `INDEX`). A
permutation campaign spans many submissions, so its launchers take
`CAMPAIGN_ROOT` (and the aggregate launcher `NORMALIZED_CAMPAIGN_ROOT` and
`ANALYSIS_DIR`); set `OUTPUT_DATE` or `CAMPAIGN_ROOT` so every pair of one
campaign lands under the same root. Launchers that read completed past runs
name them at their post-migration `outputs/YYYY-MM-DD/experiments/legacy_runs/`
paths.

The migration tool is dry-run by default

```sh
uv run python -m src.scripts.organize_outputs --before YYYY-MM-DD
```

Use `--exclude NAME` for any output still read or written by a live job.
After checking the planned paths, add `--apply`. The tool refuses to overwrite
an existing destination and writes `outputs/organization_manifest.json` after
a successful migration.

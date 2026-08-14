# src/plotting package (2026-08-14)

Figure and table scripts under `scripts/` each carried their own copy of the
same helpers, the TSFM run registry lived inside a plotting script that three
other scripts imported from, and the synthetic-toy figures sat in
`src/loss_space/plots.py`. All of it now lives in `src/plotting`, which leaves
`scripts/` for launchers and data-processing CLIs.

## Layout

`src/plotting/core/`
- `color_palette.json` plus `palette.py` hold the LTS palette. `apply_palette()`
  installs `categoricalOrder` as matplotlib's default color cycle and runs on
  package import, so every figure picks it up without any per-script setup.
  Explicit color choices now name palette entries (`PRIMARY`, `SECONDARY`,
  `TEAL`) instead of `tab:*` or the old `#0072B2` / `#D55E00` pair.
- `figures.py` has `mean_ci(values, axis=0, log_space=False)`, which derives the
  Student-t critical value from the sample size instead of the three hardcoded
  `T_CRITICAL_95_DF7` / `T_CRITICAL_95_DF14` constants, plus `save_figure`
  (writes the .pdf and .png pair, makes the parent directory, closes the
  figure), `style_step_axis`, `bottom_legend`, and `GLOBAL_STYLE`.
- `figures.py` also carries `PAPER_RCPARAMS` / `apply_paper_style()`, applied on
  package import: `pdf.fonttype`/`ps.fonttype` 42 so PDFs embed TrueType instead
  of the Type 3 fonts matplotlib defaults to and most venues reject,
  `svg.fonttype` none, 300 dpi rasters, opaque backgrounds, frameless legends,
  and hairline-safe axis and tick widths. `save_figure` crops with
  `bbox_inches="tight"` and `pad_inches=0.02`, and passes
  `metadata={"CreationDate": None}` so a rerun on unchanged data produces a byte
  identical PDF. Subfigure panels pass `bbox_inches=None` so per-panel cropping
  does not misalign their axes.
- `tsfm_runs.py` has the four-model `MODELS` registry, `COLORS`, `LABELS`,
  `LOSS_SPACES`, the history readers, and `capped_gini`.
- `loss_space.py` is the former `src/loss_space/plots.py`.

`src/plotting/scripts/` holds the entry points, run as modules
(`uv run python -m src.plotting.scripts.<name>`): the three `plot_tsfm_*`
scripts, `summarize_tsfm_paper_results`, and per-experiment builders under
`src/plotting/scripts/reproducibility/`.

## The rest of scripts/

Every remaining Python file moved out too, so `scripts/` is now only sbatch
files and submit wrappers and nothing in it is importable.

- `src/tsfm_pretraining/scripts/` took the six CLIs that wrap
  `src.tsfm_pretraining` modules (corpus audit, window index, loss-space
  aggregation, scale-free recompute, scale-free tables, metric explorer).
  `metric_explorer_template.html` moved next to the `metric_explorer` module
  that renders it.
- `src/scripts/` took the repo-level tooling, `organize_outputs.py` and
  `permutation_schedule.py`. Moving the latter out of
  `scripts/reproducibility/real_scale_swap/` removed the tree's only
  `src` to `scripts` import.

Every launcher now invokes its target as a module (`uv run python -m ...`)
rather than by file path, including the ones that used a path before the move.
The README's script section was rewritten around this layout.

## Verification

Before the palette swap, regenerating the four-model convergence figures, the
subfigures, and the inequality figures reproduced all 13 PNGs byte for byte,
which is what confirms the helper extraction was behavior-preserving. The paper
tables matched apart from three CI half widths differing in the last two
significant digits, which is float summation order inside `mean_ci`. The figures
were then regenerated on the LTS palette and the paper defaults. `pdffonts` on
the regenerated PDF reports one embedded subset CID TrueType face and no Type 3,
and two consecutive runs produce byte identical PDFs. `tests/test_scale_swap.py` passes
(10 tests) with `PYTHONPATH=.`.

## Regenerating every paper figure

All figure sources moved into the dated output layout, so the campaign roots are
no longer the flat `outputs/<name>` paths the sbatch files still name.
`aggregate_permutation_campaign.py`'s `PAIR_ONE_*_PATHS` constants were
repointed at `outputs/2026-07-17/experiments/legacy_runs/`. The commands that
regenerate the whole figure set on the LTS palette

```sh
L=outputs/2026-07-17/experiments/legacy_runs
OUT=outputs/2026-08-14/visualizations/loss_space

uv run python -m src.plotting.scripts.reproducibility.real_scale_swap.aggregate_permutation_campaign \
  --campaign-root=outputs/2026-07-29/experiments/legacy_runs/scale_swap_permutations \
  --normalized-campaign-root=outputs/2026-08-05/experiments/legacy_runs/scale_swap_permutations_normalized \
  --output-dir=$OUT/scale_swap_permutation_analysis

uv run python -m src.plotting.scripts.reproducibility.real_scale_swap.aggregate_crossover \
  --assignment-a=$L/18948_scale_swap_a --assignment-b=$L/18948_scale_swap_b \
  --lr-adjusted-a=$L/18985_scale_swap_lr_adjusted_a \
  --lr-adjusted-b=$L/18985_scale_swap_lr_adjusted_b \
  --output-dir=$OUT/scale_swap_crossover

uv run python -m src.plotting.scripts.reproducibility.synthetic_loss_space.replot_metrics \
  outputs/2026-07-29/experiments/legacy_runs/23555_loss_space_toy 200,500,2000
uv run python -m src.plotting.scripts.reproducibility.synthetic_loss_space.replot_forecasts \
  outputs/2026-07-29/experiments/legacy_runs/23555_loss_space_toy
```

with the same two synthetic commands for `23556_loss_space_toy` (Adam), the
per-dataset variance-bin aggregate over the eight `18857`/`19019` run
directories, and `replot_metrics` on
`outputs/2026-07-16/experiments/legacy_runs/18857_electricity_loss_space_toy`
for the per-bin appendix figures.

## Figure provenance changes to check

Installing the regenerated files into `overleaf/figures/loss_space/` changed the
source run behind two groups of appendix panels, because the previously
installed files predated the runs the launchers now name as canonical.

- `synthetic_loss_space/ADAM/{grad_magnitude,nmse_core_linear_200,
  nmse_controls_linear_200,nmse_global_linear_500}.pdf` were from a June run.
  They now come from job 23556, the Adam run
  `install_paper_figures.sbatch` names.
- `real_variance_bins/{nmse_core_linear_2000,grad_magnitude}.pdf` now come from
  the electricity variance-bin run 18857, matching the five-seed per-bin figure
  the appendix caption describes.

If the surrounding text quotes numbers from the older runs, revert those six
files rather than editing the text.

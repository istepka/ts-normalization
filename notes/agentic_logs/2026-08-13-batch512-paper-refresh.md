# Batch 512 TSFM paper refresh

Chronos-2 job 29436 and MOIRAI 2.0 job 29437 completed 24 runs each.
Every run used four seeds, 30,000 optimizer updates, and zero skipped updates.
The runs include natural data and complementary controlled scale assignments.

The completed runs now live under the dated experiment hierarchy.

- `outputs/2026-08-12/experiments/tsfm_pretraining/chronos2/gifteval_chronos2_b512_29436`
- `outputs/2026-08-12/experiments/tsfm_pretraining/moirai2/gifteval_moirai2_b512_29437`

`src/plotting/scripts/summarize_tsfm_paper_results.py` regenerates the three robust paper tables.
It averages each natural-data dataset across four seeds before computing median MASE, P90/P50, capped Gini, and dataset wins.
For controlled comparisons it computes one geometric mean ratio per seed and uses a 95 percent Student-t interval across seeds.

`src/plotting/scripts/plot_tsfm_natural_convergence.py` regenerates the full four-model convergence figures.
It reads MOMENT and TimesFM from the organized legacy directory and reads the new Chronos-2 and MOIRAI 2.0 runs from their dated experiment directories.
It does not generate the early-training figures.

The new natural-data median MASE values are 0.815 versus 1.560 for Chronos-2 and 0.782 versus 1.358 for MOIRAI 2.0 under normalized-space versus original-space loss.
The controlled b=1 divided by b=10 ratio under original-space loss is 1.104 for Chronos-2 and 1.119 for MOIRAI 2.0.
The corresponding normalized-space ratios are 0.998 for both models.

Generated tables are under `outputs/2026-08-13/analysis/tsfm_pretraining/four_model_loss_space`.
Generated plots are under `outputs/2026-08-13/visualizations/tsfm_pretraining/four_model_natural_convergence`.
The main results section, appendix tables, and full convergence PDFs in `overleaf/` were updated from these artifacts.

# 03 — Natural-variance GIFT-Eval subset (deferred, confounded)

**Status:** planned, secondary. Lower priority than notes 01–02 because it is
confounded; use it as supporting evidence, not the primary claim.

## Goal

Show the loss-space disparity on a heterogeneous real-world corpus — the setting that
actually motivates the paper (multi-dataset / cross-frequency TSFM pretraining).

## Design

Pick a subset of GIFT-Eval datasets whose variances span different scales. Stratify
(or trim to equal length and sample uniformly) so each dataset contributes equally per
batch. Train the shared model under normalized vs original loss space; log per-dataset
loss curves, per-dataset gradient magnitude, per-dataset and global nMSE.

## Confounds (why this is secondary)

Real datasets differ in more than variance: signal complexity, SNR, seasonality,
horizon difficulty. So convergence speed correlating with variance is *suggestive* but
not conclusive. The `experiments.tex` idea of "normalizing convergence speed by SNR" is
hand-wavy and hard to defend. Prefer note 02's constant-rescaling design for the clean
causal claim, and use this corpus-level experiment to show the effect still appears in
the wild.

## Dropped variant

Correlating *pretrained-TSFM* zero-shot per-dataset test MSE with dataset variance does
**not** test the loss-space training claim: those models were already trained in
normalized space (RevIN/instance norm) and denormalize only for evaluation, so the
correlation reflects eval-metric scaling, not training-time gradient bias.

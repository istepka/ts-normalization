# 02 — Constant-rescaled real data (deferred)

**Status:** planned, not yet implemented. Build after the synthetic toy (note 01) is
solid.

## Goal

Reproduce the loss-space disparity on *real* signals while isolating variance from
confounds. Natural datasets differ in variance but also in signal complexity, SNR, and
seasonality, so a raw cross-dataset comparison cannot attribute convergence differences
to variance alone (see note 03).

## Design

Take a few real univariate series (e.g. ECL, solar, a Favorita slice). For each, create
scaled copies by multiplying the *whole series* by a constant factor `c ∈ {1, 10, 100}`.
Constant rescaling changes variance but **not** the normalized pattern, SNR, or
seasonality — so any per-copy convergence difference under original-space loss is purely
the `b^2 = σ^2` gradient scaling, exactly mirroring the synthetic toy on real data.

- Treat each (dataset, scale) pair as a "category"; stratify batches across them.
- Reuse the toy's machinery: instance norm (RevIN), shared model, normalized vs
  original loss, SGD (Adam would erase the effect — it is per-coordinate scale
  invariant), grad-norm-matched and equal-scale controls.
- Metric: per-category nMSE in normalized space.

## Expected result

Original-space loss → the ×100 copy dominates the shared gradient and converges first
(or destabilizes), the ×1 copy lags; normalized-space → all copies converge together.
Equal-scale control removes the disparity; grad-norm-matched control preserves it.

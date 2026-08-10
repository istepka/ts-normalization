# Chronos-2 loss-space integration

## Purpose

Chronos-2 adds a second forecasting architecture to the loss-space study. It
tests whether the TimesFM result is specific to TimesFM or common across
forecasting models whose targets lie outside the input context.

This is a pretraining experiment from random initialization. It does not use
the published Chronos-2 weights and it does not claim to reproduce the
published Chronos-2 training run.

## Model

The implementation uses `chronos-forecasting==2.3.1`. The configured model
uses the official Chronos-2 Small dimensions and has 27,934,624 parameters.
It has six layers, hidden size 512, eight attention heads, patch size 16, and
13 output quantiles. It uses eager attention so deterministic CUDA mode does
not silently select a hardware-dependent attention kernel.

The experiment uses 512 context points and predicts the next 128 points. The
model keeps Chronos-2's context-based standardization and arcsinh transform.

Official sources are listed below.

- https://github.com/amazon-science/chronos-forecasting
- https://huggingface.co/autogluon/chronos-2-small

## Loss comparison

`chronos2_normalized` is the native Chronos-2 objective. The future target is
normalized with statistics from the context. Quantile loss is then computed
in that normalized space.

`chronos2_original` uses the same normalized predictions from the same forward
pass. It reverses the normalization and computes the same quantile loss
against the target in its original units.

Nothing else changes between the two conditions. They share initialization,
window order, optimizer, learning rate, training length, and evaluation data.

The controlled scale experiment predicts a linear effect. Multiplying a
dataset by ten should multiply the original-space quantile-loss gradient by
about ten. It should not change the normalized-space gradient.

## Evaluation

The model uses the existing GiftEvalPretrain window index and natural-scale
evaluation. It reports per-window nMSE and MASE. It also reports their
per-dataset values and Gini coefficients at every evaluation checkpoint.

nMSE uses squared point-forecast error divided by the squared context scale.
The point forecast is the predicted median. MASE uses the same median forecast
and the existing seasonal-naive denominator.

## Replication

The launcher defaults to seeds 0, 1, 2, and 3. It contains no fifth seed. Each
seed runs the following six conditions.

1. Normalized loss on natural scales
2. Original loss on natural scales
3. Normalized loss with controlled scale assignment A
4. Normalized loss with controlled scale assignment B
5. Original loss with controlled scale assignment A
6. Original loss with controlled scale assignment B

The Slurm array is capped at eight concurrent GPU jobs. The launcher has been
prepared but no jobs were submitted during implementation.

## Verification

The focused tests check the following properties.

- The mean per-example native loss matches the official model loss.
- Original predictions are the exact inverse transform of normalized
  predictions.
- A tenfold scale change leaves the normalized gradient ratio near one.
- A tenfold scale change makes the original quantile-loss gradient ratio near
  ten.
- The training loop writes checkpoints and summaries through the shared entry
  point.

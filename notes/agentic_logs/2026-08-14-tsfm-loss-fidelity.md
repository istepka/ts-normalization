# Each TSFM trains on its own paper's objective (2026-08-14)

Collecting the four duplicated quantile losses into `src/losses/quantile.py`
(see [2026-08-14-src-restructure.md](2026-08-14-src-restructure.md)) put the
two conventions side by side, which made it visible that Moirai-2.0 was not
training on its own. Each adapter was then checked against upstream source
rather than against memory of the paper.

## Chronos-2, already correct

`chronos/chronos2/model.py:551` in the installed `chronos-forecasting==2.3.1`
computes `2 * |(target - pred) * ((target <= pred) - q)|` and reduces it with
`.mean(dim=-1).sum(dim=-1)`, its own comment reading "mean over prediction
horizon, sum over quantile levels and mean over batch". `crps_quantile_loss`
matches this exactly. The factor of 2 and the sum over levels are genuinely
Chronos-2's, not an error.

## Moirai-2.0, training on 18x its objective

`moirai2_adapter.py` had copied Chronos-2's loss verbatim, which is why the two
`_quantile_loss` definitions were byte-identical. uni2ts instead trains Moirai
2.0 with `PackedQuantileMAELoss`, whose `_loss_func` ends in
`quantile_loss.mean(dim=-2)`: the plain pinball loss averaged over quantile
levels, with no factor of 2. With `conf/tsfm_moirai2.yaml`'s 9 levels (the
upstream default) the adapter was optimizing 2Q = 18 times the paper's
objective. Measured directly, the ratio is 18.0000.

Now calls `pinball_loss` with the horizon mask, checked against a transcription
of uni2ts over 200 random trials.

This is not merely a rescaled loss number. `grad_clip_norm` is 1.0 and
clipping is not scale invariant, so an 18x gradient crossed the threshold far
more often than it should have, which is exactly the quantity this project
measures.

## TimesFM, our own composition rather than the paper's

The vendored `pytorch_patched_decoder.py` is Google's inference-only v1 port
and carries no training loss, since v1 training code was never released, so
`objective: combined` (MSE plus pinball over the quantile heads) was this
project's invention. The paper trains the point forecast with MSE.
`conf/tsfm_timesfm.yaml` now selects `objective: mse`, an option the adapter
already supported.

Consequence worth remembering: under `mse` only the point slice of
`horizon_ff_layer` enters the loss, so the head's quantile outputs receive no
gradient and mean nothing in such a run. Set `objective: combined` if the
quantile outputs are ever needed.

## MOMENT

Masked reconstruction MSE, matching its pretraining objective. Unchanged.

## Existing runs predate both changes

Neither model was rerun. The pretraining corpus composition is expected to
change, at which point everything is retrained anyway. Until then, treat these
as stale:

- every Moirai-2.0 run, including `gifteval_moirai2_b512_29437`
- every TimesFM run, including `timesfm_natural_eval250_30192` and the 60k
  continuation

The paired normalized-vs-original contrast within each model still holds, since
both conditions always shared one loss. What is affected is each model's
absolute loss curve, its convergence rate against the other three, and anything
read off gradient magnitude or clipping frequency.

## Guardrail

`tests/test_losses.py` pins `pinball_loss` against a transcription of uni2ts
and `crps_quantile_loss` against one of Chronos-2, plus a third test asserting
the two differ by exactly 2Q. Merging them back into one function, or swapping
which adapter calls which, now fails the suite.

# Variable-context pretraining windows

Status: proposal, not implemented. Written 2026-08-17 after the first
held-out evaluation of a real checkpoint.

## The evidence that prompted this

The 512-context window is not a tuning choice. It decides which datasets
exist at all, and the first real evaluation showed what that costs.

**Most of the corpus is invisible.** The window index at context 512 plus
horizon 128 requires 640 valid points per window. It holds 40,952,582 windows
over **53 datasets**. The corpus root has **152**. So 99 dataset directories
contribute nothing, not because they were excluded but because no series in
them is long enough.

**The held-out benchmarks are mostly unreachable too.** Train-split lengths
against the 640 points a window needs:

| suite | series | max | median | reaching 640 | windows possible |
|---|---|---|---|---|---|
| M1 | 1,001 | 132 | 53 | 0 | 0 |
| M3 | 3,003 | 126 | 51 | 0 | 0 |
| Tourism | 1,311 | 309 | 102 | 0 | 0 |
| M4 | 100,000 | 9,919 | 97 | 4,087 | 17,348 |

This answers a question that came up directly: adding the M-competition train
splits to the pretraining corpus would change nothing. M1, M3, and Tourism
cannot produce a single window, and M4 would add 17,348, which is 0.04% of
the index, drawn only from its 4% longest series.

**And it shows up in the results.** Chronos-2 at 60k steps, native protocol,
MASE, four seeds:

| suite | SIT | RevIN | seasonal naive |
|---|---|---|---|
| favorita | **0.838** | 0.889 | 1.079 |
| gifteval | **1.603** | 2.588 | 1.824 |
| m1 | 2.304 | 3.063 | **2.117** |
| m3 | 1.859 | 5.463 | **1.764** |
| m4 | 2.186 | 4.463 | **2.057** |
| tourism | 3.260 | 4.052 | **2.412** |

The pretrained model beats seasonal naive on exactly the two suites whose
series are long, and loses on the four whose series are short. A median M1
series is 53 points, so the eval left-pads it into a 512-wide context and
roughly 90% of what the model sees is padding it never saw in training. This
is a context-length train/test mismatch, not a data-coverage gap.

Note this hits SIT and RevIN identically, so the loss-space comparison the
paper is about is unaffected. What it limits is the absolute claim.

## Proposal

Admit a window when a series has at least **64 context points and 8 horizon
points**, and take up to **512 context and 128 horizon** wherever the series
allows it. Long series keep exactly the windows they have now.

## What it requires

**A rebuilt index.** `build_gifteval_window_index.py` currently emits one
fixed geometry. It would need to emit per-window context and horizon lengths,
and `WindowIndex` would need to carry them.

**Ragged batching.** Windows of differing length cannot stack into one
tensor. Two options:
- Pad to the batch maximum and mask. The adapters already take a validity
  mask, so this is the smaller change, but it wastes compute when a batch
  mixes 64-point and 512-point windows.
- Bucket by length so each batch is close to uniform. More efficient and more
  code, and it correlates batch composition with series length, which the
  schedule seed then has to account for.

**A sampling rule.** Short windows will vastly outnumber long ones once the
99 absent datasets appear. Left alone they would dominate every batch, which
trades the current bias for its mirror image. Whatever rule is chosen has to
be recorded per run, because it becomes part of what a checkpoint means.

## Open questions

- Does the horizon stay proportional to the context, or is it fixed at 8 for
  short windows? A 64-point context with a 128-point horizon is not a
  forecasting problem the model can learn from.
- TimesFM's `prediction_length` must stay 128 to match every TimesFM config's
  `horizon_len`. A variable horizon may not be expressible for that adapter
  without changing the config contract.
- Moirai 2.0's binding native horizon is 64, already below the 128 the index
  builds. Variable horizons interact with that.
- Does the eval harness change? The `native` mode already forecasts from
  whatever history a series has, so it needs nothing. `fixed` mode is defined
  as the training window shape, so its definition would follow the new
  geometry.

## Cost of the change

Rebuilding the index moves `_dataset_scale_group`, which splits the
stable-hash-ordered dataset list at `len // 2`. Going from 53 datasets to
something near 152 reassigns most datasets to a different group, so
controlled-scale results from before and after cannot be pooled. The
2026-08-14 rebuild already moved one dataset for this reason. Natural-scale
runs are unaffected.

Every completed run is at context 512, so adopting this means the comparison
set restarts rather than extends.

## Not yet measured

How many series and windows a 64 plus 8 index actually admits, per dataset.
An inline probe over 99 dataset directories was too slow to finish
interactively. This wants a short CPU job that walks the corpus once and
reports the window count at several geometries, so the tradeoff can be sized
before any of the above is built.

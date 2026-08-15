# Normalization as a module, not a backbone internal (2026-08-14)

Five places in this repo hand-rolled the same fork: normalize the input with
per-window statistics, then decide whether the loss is taken in normalized or
original space. `PatchTransformer.normalize` plus its trainer, and each of the
four TSFM adapters reading statistics out of its own backbone.

`src/models/normalization.py` makes that one object. It does not touch the loss.

## Interface

    z_context, transform_stats = norm.transform_input(context, valid)
    output = backbone(z_context)
    output, target = norm.transform_target_and_output(output, target, transform_stats)

`SIT` moves the target into normalized space. `RevIN` moves the output back
into original space. Both put the two tensors into a common space and stop
there, so the same three lines work for every model and every objective.
`align_target_output` is an alias for the second call.

`TransformStats` is returned rather than stored on the module. Stashing it on
`self` would cross contaminate under gradient accumulation, evaluation
interleaved with training, and DDP, and it would hide the dependency between
the two calls. It carries `forward`/`inverse` rather than a bare `(loc, scale)`
pair because one scheme is not affine, see below.

`apply_causal_norm` takes the statistics from `extra_context`, a strictly
earlier window, so the context itself never enters them. Setting it without
passing `extra_context` raises rather than falling back. Default is off, since
nothing trains causally yet.

## Schemes are registered because they genuinely differ

Epsilon placement, Bessel correction, and NaN handling all vary between
upstream implementations, so one parameterized standard scaler would have been
wrong for at least three of the four models. `SCHEMES` holds `std`,
`moment_revin`, `chronos2`, `moirai2_std`, `moirai2_absmean`, and `timesfm`.

The differences worth naming:

- `moirai2_std` adds `minimum_scale` to the variance *under* the square root,
  so its smallest reportable scale is `sqrt(minimum_scale)`, not
  `minimum_scale`.
- `moment_revin` uses the uncorrected (population) variance via `nanmean`,
  while `std` is Bessel corrected.
- `timesfm` has two modes. `first_patch` takes statistics from the first patch
  with more than three unpadded values, which is upstream. `whole_context` is
  this repo's causal variant.

## Chronos-2 is invertible but not affine

`InstanceNorm` standardizes and then applies `arcsinh`. The arcsinh is not a
loc/scale operation, so "disable the backbone's normalization" has no clean
meaning for this model. Injecting `loc=0, scale=1` would leave arcsinh acting
on raw magnitudes, crushing exactly the scale variation this project measures.

So arcsinh moved into the scheme and the scheme is named for what it is. It is
fully reversible, `sinh` is its exact inverse, and `inverse` casts to float32
first because sinh overflows fast in reduced precision.

The consequence is not a bug but it belongs in the paper's model table. For
MOMENT, Moirai-2.0, and TimesFM the normalized and original arms are related by
an exact affine map, so their losses differ by a per-window constant. For
Chronos-2 they are related nonlinearly, and the effective per-sample weighting
depends on where on the arcsinh curve the error sits.

## MOMENT normalizes over visible positions only

Found while writing the identity test, which failed on this model alone.
`vendor/moment/models/moment.py:303` computes the RevIN statistics from
`mask * input_mask`, the randomly drawn pretrain mask intersected with the
input mask, which keeps the positions being reconstructed out of them.

MOMENT's normalization is therefore not a pure function of the context. It
depends on a random mask drawn inside the forward pass. The other three have no
such coupling.

The adapter handles it. `forward` already owned the RNG fork, so the mask draw
moved up one line and goes back in through `forward`'s `mask` argument:

    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(batch.batch_seed)
        pretrain_mask = model.mask_generator.generate_mask(...)
        z_context, stats = module.transform_input(
            batch.x_enc.squeeze(1), pretrain_mask * batch.input_mask
        )
        out = model(x_enc=..., input_mask=..., mask=pretrain_mask)

Drawing the mask first, inside the same fork and before any other RNG consumer,
leaves the stream exactly where the model's own internal draw would have left
it. The identity test runs in `train()` mode so dropout draws too, and asserts
the externally drawn mask equals the one the unmodified model produces.

## Disabling the backbones

`src/models/norm_shims.py` neutralizes each backbone's built-in normalization
without editing any vendored file, so the pinned upstream source stays
byte-identical to its `REVISION`.

| model | mechanism |
| --- | --- |
| Moirai-2.0 | swap `model.scaler` for upstream's own `PackedNOPScaler` |
| Chronos-2 | swap `model.instance_norm` for an identity with the same `inverse` |
| MOMENT | swap `model.normalizer` for an identity RevIN exposing `mean`/`stdev` |
| TimesFM | rebind `_forward_transform`/`_reverse_transform` on the instance |

TimesFM is the odd one because its normalization is two methods rather than a
submodule, so there is no attribute to swap.

Each shim keeps the attributes the adapters read off the normalizer it replaces
(`eps`, `minimum_scale`, `mean`, `stdev`), so disabling normalization does not
drag the surrounding metric code with it.

## Verification

`tests/test_normalization.py` pins every scheme against its upstream reference,
and `tests/test_norm_shims.py` runs each seeded model twice, once unmodified and
once with its normalization disabled and the scheme applied externally.

All four backbones reproduce **bit-identically** (`torch.equal`) in both the
normalized and the original space. That is the gate: the module can replace a
backbone's internal normalization without moving any training numerics, so
adopting it does not invalidate existing runs.

20 new tests, all passing.

## Adopted so far

MOMENT only. `build_moment_model` now disables the backbone's normalizer, and
`forward` picks `SIT` or `RevIN` off the condition and takes the loss through
`transform_target_and_output`, so the condition fork is one call instead of a
branch on a metric name. The reported `normalized_mse` and `original_mse` are
unchanged, they are just derived from `stats` rather than from the model's
`revin_mean`/`revin_stdev` metadata, which is now the identity.

Chronos-2, Moirai-2.0, and TimesFM still call their backbones' normalization
directly. Rewiring them is the follow-up, one model per commit, each gated on
the identity test that already exists for it. The recipe is proven, the work is
mechanical.

`src/training/loss_space.py` also still calls `PatchTransformer.normalize` and
`normalize_target` directly. `StandardScheme` is pinned equal to it, but that
path produces published figures, so it was left alone rather than touched
without a reason to re-run.

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

## Nothing in normalization.py knows about a backbone

The first cut of this module failed that test. It held six schemes, four of
them named after the model they came from, three importing or mirroring
vendored code, and a `SCHEMES` name registry that had to see all of them. A
separate `norm_shims.py` then collected one disable function per backbone in a
second shared place. Backbone knowledge in the generic layer, twice over.

The schemes were misnamed rather than misplaced. Masked uncorrected std, a
variance floor under the square root, mean absolute scaling, and standardize
then arcsinh are all general statistics. They are now named for what they
compute:

| scheme | statistic | reproduces |
| --- | --- | --- |
| `StandardScheme` | corrected std, eps on the scale | `PatchTransformer` |
| `PopulationStdScheme` | uncorrected std, NaN reduction | MOMENT's RevIN |
| `FlooredStdScheme` | corrected std, floor under the sqrt | `PackedStdScaler` |
| `AbsMeanScheme` | zero loc, mean absolute scale | `PackedAbsMeanScaler` |
| `ArcsinhStdScheme` | standardize, then arcsinh | Chronos-2 `InstanceNorm` |

The three std variants were kept separate rather than folded into one class
with `correction` and `eps_placement` flags. They differ on all-invalid
windows, where the NaN reduction yields NaN and the masked form yields clamped
values, so the flags would have hidden three real behaviors behind config.

`SCHEMES` and `build_scheme` were deleted. Nothing called them, and a
name-to-class map only works if it can see every scheme, which reintroduces
exactly the coupling this split removes. Scheme choice per backbone is fixed by
what reproduces upstream, and it is not the axis this paper varies. That axis
is `SIT` vs `RevIN`, already a config knob.

## The prototype a new baseline subclasses

`BackboneNormalization` is what one backbone contributes to the contrast, and
adding a baseline means subclassing it in that model's adapter module:

    class MomentNormalization(normalization.BackboneNormalization):
        normalized_condition = "moment_normalized"
        original_condition = "moment_original"

        def __init__(self):
            super().__init__(normalization.PopulationStdScheme(eps=1e-5))

        def disable(self, model):
            model.normalizer = IdentityRevIN(eps=model.normalizer.eps)

Three things per model: the scheme, the two condition names, and how to silence
the backbone. `module(condition)` then returns the `SIT` or `RevIN` that
condition asks for and raises on anything else, so each adapter's `forward` is
one call instead of a hand-rolled branch plus its own condition check.

Conditions are named rather than positional because the adapters do not agree
on an order. TimesFM's are `("timesfm_native_original", "timesfm_normalized")`,
reversed relative to the other three. `conditions()` is a classmethod because
three of the four schemes need the model config to construct, while the
training loop validates `cfg.condition` before it has one.

`disable` lives on the subclass because it is the one genuinely
backbone-specific piece. It reaches into a particular attribute or method of a
particular vendored class:

| model | mechanism |
| --- | --- |
| Moirai-2.0 | swap `model.scaler` for upstream's own `PackedNOPScaler` |
| Chronos-2 | swap `model.instance_norm` for an identity with the same `inverse` |
| MOMENT | swap `model.normalizer` for an identity RevIN exposing `mean`/`stdev` |
| TimesFM | rebind `_forward_transform`/`_reverse_transform` on the instance |

None of them edit vendored source, so the pinned upstream files stay
byte-identical to their `REVISION` (`git diff` over `src/models/vendor/` is
empty). TimesFM is the odd one because its normalization is two methods rather
than a submodule, so there is no attribute to swap. Each replacement keeps the
attributes the adapters read off the normalizer it replaces (`eps`,
`minimum_scale`, `mean`, `stdev`), so disabling normalization does not drag the
surrounding metric code with it.

`TimesFMScheme` stayed in `timesfm.py` rather than moving to the generic
module. It is the one scheme genuinely entangled with its backbone, since
TimesFM carries padding in-band as the sentinel `1123581321.0` and `forward`
has to re-stamp it after scaling. That is a data encoding, not a statistic.

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

## Verification

Tests follow the code. `tests/test_normalization.py` covers the machinery every
model shares, meaning statistics threading, the SIT/RevIN contrast, causal
statistics, and the `BackboneNormalization` prototype. Each adapter's test file
carries its own two normalization tests, since the upstream reference it is
pinned against lives there:

- the scheme reproduces the backbone's own normalizer exactly
- the backbone with its normalization disabled and the scheme applied
  externally reproduces the unmodified backbone

All four reproduce **bit-identically** (`torch.equal`) in both the normalized
and the original space. That is the gate: the module can replace a backbone's
internal normalization without moving any training numerics, so adopting it
does not invalidate existing runs.

Full suite: 181 passed.

## Adopted so far

MOMENT only. `build_moment_model` disables the backbone's normalizer, and
`forward` takes its module from `NORMALIZATION.module(condition)` and its loss
through `transform_target_and_output`. The reported `normalized_mse` and
`original_mse` are unchanged, just derived from `stats` rather than from the
model's `revin_mean`/`revin_stdev` metadata, which is now the identity.

Chronos-2, Moirai-2.0, and TimesFM now have their `BackboneNormalization`
subclass and a passing identity test, but their `forward` still calls the
backbone's normalization directly. Rewiring them is the follow-up, one model
per commit. The recipe is proven and the work is mechanical.

`src/training/loss_space.py` also still calls `PatchTransformer.normalize` and
`normalize_target` directly. `StandardScheme` is pinned equal to it, but that
path produces published figures, so it was left alone rather than touched
without a reason to re-run.

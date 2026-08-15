"""The module plus a disabled backbone must reproduce the backbone exactly.

This is the gate on adopting `src/models/normalization.py` for the four TSFM
adapters. Each test runs the same seeded model twice: once unmodified, once
with its internal normalization disabled and the equivalent scheme applied
externally. Bit-identical output means the refactor moves no training numerics
and leaves existing runs interpretable.
"""

import dataclasses

import torch

from src.data.gifteval import window_index as wi
from src.models import norm_shims
from src.models import normalization as norm
from src.models.vendor.timesfm_v1.pytorch_patched_decoder import TimesFMConfig


def _index(tiny_corpus, context_length, prediction_length):
    root, domain_map = tiny_corpus
    config = wi.WindowIndexConfig(
        context_length=context_length,
        prediction_length=prediction_length,
        stride=80,
        val_series_fraction=0.25,
        base_seed=0,
    )
    index = wi.build_window_index(root, ["synth_a", "synth_b"], domain_map, config)
    return index, wi.SeriesCache(root)


def test_timesfm_scheme_reproduces_the_backbone(tiny_corpus):
    from src.models import timesfm as tf

    config = TimesFMConfig(
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        hidden_size=32,
        intermediate_size=32,
        head_dim=8,
        patch_len=32,
        horizon_len=32,
    )
    index, cache = _index(tiny_corpus, 64, config.horizon_len)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = tf.make_batch(index, rows, cache, config.horizon_len)

    reference = tf.build_timesfm_model(config, seed=0)
    reference.eval()
    with torch.no_grad():
        normalized_reference, original_reference, _ = tf.run_decoder(reference, batch)

    shimmed = tf.build_timesfm_model(config, seed=0)
    norm_shims.disable_timesfm_normalization(shimmed)
    shimmed.eval()

    scheme = norm.TimesFMScheme(
        mode="first_patch",
        patch_len=config.patch_len,
        tolerance=config.tolerance,
        pad_val=config.pad_val,
    )
    module = norm.SIT(scheme)
    valid = 1.0 - batch.context_padding
    z_context, stats = module.transform_input(batch.context, valid)
    with torch.no_grad():
        normalized_shimmed, _, _ = tf.run_decoder(
            shimmed, dataclasses.replace(batch, context=z_context)
        )

    assert torch.equal(normalized_reference, normalized_shimmed)
    # RevIN reads the same forward pass in the original space.
    assert torch.equal(stats.inverse(normalized_shimmed), original_reference)


def test_chronos2_scheme_reproduces_the_backbone(tiny_corpus):
    from src.models import chronos2 as c2

    config = c2.Chronos2Config(
        context_length=64,
        prediction_length=32,
        patch_size=16,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.0,
        initializer_factor=0.05,
        quantiles=(0.1, 0.5, 0.9),
        use_arcsinh=True,
        grad_clip_norm=1.0,
    )
    index, cache = _index(tiny_corpus, 64, 32)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = c2.make_batch(index, rows, cache)

    reference = c2.build_chronos2_model(config, seed=0)
    reference.eval()
    with torch.no_grad():
        normalized_reference, original_reference, _ = c2.run_model(reference, batch)

    shimmed = c2.build_chronos2_model(config, seed=0)
    norm_shims.disable_chronos2_normalization(shimmed)
    shimmed.eval()

    scheme = norm.Chronos2Scheme(eps=1e-5, use_arcsinh=config.use_arcsinh)
    module = norm.SIT(scheme)
    z_context, stats = module.transform_input(batch.context, batch.context_valid)
    with torch.no_grad():
        normalized_shimmed, _, _ = c2.run_model(
            shimmed, dataclasses.replace(batch, context=z_context)
        )

    assert torch.equal(normalized_reference, normalized_shimmed)
    assert torch.equal(stats.inverse(normalized_shimmed), original_reference)


def test_moirai2_scheme_reproduces_the_backbone(tiny_corpus):
    from src.models import moirai2 as m2

    config = m2.Moirai2Config(
        context_length=64,
        predict_horizon=32,
        patch_size=16,
        d_model=64,
        d_ff=64,
        num_layers=2,
        max_seq_len=64,
        attn_dropout_p=0.0,
        dropout_p=0.0,
        scaling=True,
        quantile_levels=(0.1, 0.5, 0.9),
        grad_clip_norm=1.0,
    )
    index, cache = _index(tiny_corpus, 64, 32)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = m2.make_batch(index, rows, cache, config)

    reference = m2.build_moirai2_model(config, seed=0)
    reference.eval()
    with torch.no_grad():
        preds_reference, target_reference, loc, scale = m2.run_model(
            reference, batch, config
        )

    shimmed = m2.build_moirai2_model(config, seed=0)
    norm_shims.disable_moirai2_normalization(shimmed)
    shimmed.eval()

    scheme = norm.Moirai2StdScheme(
        correction=1, minimum_scale=reference.scaler.minimum_scale
    )
    module = norm.SIT(scheme)
    batch_size = batch.target.shape[0]
    context_patches = config.context_token_length
    context = batch.target[:, :context_patches].reshape(batch_size, -1)
    context_valid = batch.observed_mask[:, :context_patches].reshape(batch_size, -1)
    _, stats = module.transform_input(context, context_valid.float())

    assert torch.equal(stats.loc, loc)
    assert torch.equal(stats.scale, scale)

    sequence = batch.target.reshape(batch_size, -1)
    normalized_sequence = stats.forward(sequence).view_as(batch.target)
    with torch.no_grad():
        preds_shimmed, target_shimmed, _, _ = m2.run_model(
            shimmed, dataclasses.replace(batch, target=normalized_sequence), config
        )

    assert torch.equal(preds_reference, preds_shimmed)
    assert torch.equal(target_reference, target_shimmed)


def test_moment_scheme_reproduces_the_backbone(tiny_corpus):
    """MOMENT normalizes over visible positions only, so the mask comes first.

    `vendor/moment/models/moment.py:303` derives the RevIN statistics from
    `mask * input_mask`, the randomly drawn pretrain mask intersected with the
    input mask, which keeps the positions being reconstructed out of the
    statistics. Hoisting normalization out of this backbone therefore means
    hoisting the mask draw with it and passing the mask back in.
    """
    from src.models import moment as mm

    config = mm.MomentConfig(
        context_length=32,
        patch_len=8,
        d_model=16,
        t5_num_layers=2,
        t5_num_heads=4,
        t5_d_ff=32,
        t5_d_kv=4,
        mask_ratio=0.3,
    )
    index, cache = _index(tiny_corpus, 32, 8)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = mm.make_batch(index, rows, cache)

    # build_moment_model now disables the backbone's normalizer, so the
    # pre-refactor reference is reconstructed by putting a real RevIN back.
    from src.models.vendor.moment.models.layers.revin import RevIN

    reference = mm.build_moment_model(config, seed=0)
    reference.normalizer = RevIN(num_features=1, eps=1e-5, affine=False)
    reference.train()
    # train() mode so dropout draws too: the mask must come off the same
    # stream position it would have inside the forward pass.
    with torch.random.fork_rng():
        torch.manual_seed(batch.batch_seed)
        with torch.no_grad():
            out_reference = reference(x_enc=batch.x_enc, input_mask=batch.input_mask)
    pretrain_mask = out_reference.pretrain_mask

    shimmed = mm.build_moment_model(config, seed=0)
    shimmed.train()

    module = norm.SIT(norm.MomentRevINScheme(eps=1e-5))
    with torch.random.fork_rng():
        torch.manual_seed(batch.batch_seed)
        drawn_mask = shimmed.mask_generator.generate_mask(
            x=batch.x_enc, input_mask=batch.input_mask
        )
        z_context, stats = module.transform_input(
            batch.x_enc.squeeze(1), drawn_mask * batch.input_mask
        )
        with torch.no_grad():
            out_shimmed = shimmed(
                x_enc=z_context.unsqueeze(1),
                input_mask=batch.input_mask,
                mask=drawn_mask,
            )

    # Drawing the mask outside the forward pass must not move the RNG stream.
    assert torch.equal(drawn_mask, pretrain_mask)
    assert torch.equal(
        out_reference.metadata["normalized_reconstruction"],
        out_shimmed.metadata["normalized_reconstruction"],
    )
    assert torch.equal(
        stats.inverse(out_shimmed.metadata["normalized_reconstruction"]),
        out_reference.reconstruction,
    )


def test_disabled_normalizers_are_the_identity():
    x = torch.randn(4, 1, 32) * 50.0

    revin = norm_shims.IdentityRevIN()
    assert torch.equal(revin(x, mode="norm"), x)
    assert torch.equal(revin(x, mode="denorm"), x)

    instance_norm = norm_shims.IdentityInstanceNorm()
    normalized, (loc, scale) = instance_norm(x)
    assert torch.equal(normalized, x)
    assert torch.equal(loc, torch.zeros_like(loc))
    assert torch.equal(scale, torch.ones_like(scale))
    assert torch.equal(instance_norm.inverse(x, (loc, scale)), x)


def test_moirai2_shim_keeps_the_degenerate_floor():
    from src.models import moirai2 as m2

    config = m2.Moirai2Config(
        context_length=64,
        predict_horizon=32,
        patch_size=16,
        d_model=64,
        d_ff=64,
        num_layers=2,
        max_seq_len=64,
        attn_dropout_p=0.0,
        dropout_p=0.0,
        scaling=True,
        quantile_levels=(0.1, 0.5, 0.9),
        grad_clip_norm=1.0,
    )
    model = m2.build_moirai2_model(config, seed=0)
    minimum_scale = model.scaler.minimum_scale
    norm_shims.disable_moirai2_normalization(model)

    assert model.scaler.minimum_scale == minimum_scale
    target = torch.randn(2, 6, 16)
    observed = torch.ones_like(target, dtype=torch.bool)
    sample_id = torch.ones(2, 6, dtype=torch.long)
    variate_id = torch.zeros(2, 6, dtype=torch.long)
    loc, scale = model.scaler(target, observed, sample_id, variate_id)
    assert torch.equal(loc, torch.zeros_like(loc))
    assert torch.equal(scale, torch.ones_like(scale))

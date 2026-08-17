import numpy as np
import torch

from src.data.gifteval import window_index as wi
from src.models import moment as ma
from src.models import normalization as norm


def _tiny_model_config(context_length=32, patch_len=8):
    return ma.MomentConfig(
        context_length=context_length,
        patch_len=patch_len,
        d_model=16,
        t5_num_layers=2,
        t5_num_heads=4,
        t5_d_ff=32,
        t5_d_kv=4,
        mask_ratio=0.3,
    )


def _tiny_index(tiny_corpus):
    root, domain_map = tiny_corpus
    config = wi.WindowIndexConfig(
        context_length=32,
        prediction_length=8,
        stride=40,
        val_series_fraction=0.25,
        base_seed=0,
    )
    index = wi.build_window_index(root, ["synth_a", "synth_b"], domain_map, config)
    return index, wi.SeriesCache(root)


def test_masked_reconstruction_shapes(tiny_corpus):
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=6, random_state=0)
    batch = ma.make_batch(index, rows, cache)
    model = ma.build_moment_model(_tiny_model_config(), seed=0)

    result = ma.forward(model, batch, "moment_original")
    assert result.reconstruction.shape == batch.x_enc.shape
    assert result.pretrain_mask.shape == batch.input_mask.shape
    assert result.per_example_loss_masked.shape == (6,)
    assert result.per_example_loss_unmasked.shape == (6,)


def test_normalized_loss_matches_manual_recomputation(tiny_corpus):
    """The normalized loss the adapter reports must equal manually normalizing
    the model's own reconstruction/target with the same per-window statistics,
    i.e. moment_normalized is genuinely computed pre-inverse-transform, not a
    rescaled copy of the original-space loss."""
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=1)
    batch = ma.make_batch(index, rows, cache)
    model = ma.build_moment_model(_tiny_model_config(), seed=0)

    result = ma.forward(model, batch, "moment_normalized")

    # Recomputed independently of the adapter. MOMENT takes its statistics
    # from the visible positions only, so the mask is drawn before they are.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(batch.batch_seed)
        pretrain_mask = model.mask_generator.generate_mask(
            x=batch.x_enc, input_mask=batch.input_mask
        )
        loc, scale = ma.NORMALIZATION.scheme.statistics(
            batch.x_enc.squeeze(1), pretrain_mask * batch.input_mask
        )
        normalized = (batch.x_enc - loc.view(-1, 1, 1)) / scale.view(-1, 1, 1)
        out = model(x_enc=normalized, input_mask=batch.input_mask, mask=pretrain_mask)

    manual_diff = (out.metadata["normalized_reconstruction"] - normalized) ** 2
    train_mask = (1.0 - out.pretrain_mask) * batch.input_mask
    manual_mse = (manual_diff.squeeze(1) * train_mask).sum(dim=1) / train_mask.sum(
        dim=1
    ).clamp_min(1.0)

    assert torch.allclose(result.per_example_loss_masked, manual_mse, atol=1e-5)


def test_forward_pass_identical_across_conditions(tiny_corpus):
    """Only the loss space should differ between moment_normalized and
    moment_original -- the reconstruction and mask must be identical given
    the same batch (same batch_seed drives the same mask)."""
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=2)
    batch = ma.make_batch(index, rows, cache)
    model = ma.build_moment_model(_tiny_model_config(), seed=0)

    out_norm = ma.forward(model, batch, "moment_normalized")
    out_orig = ma.forward(model, batch, "moment_original")
    assert torch.equal(out_norm.pretrain_mask, out_orig.pretrain_mask)
    assert torch.allclose(out_norm.reconstruction, out_orig.reconstruction)


def test_controlled_scale_gradient_ratio_matches_b_squared():
    model = ma.build_moment_model(_tiny_model_config(), seed=0)

    torch.manual_seed(0)
    base = torch.randn(4, 32) * 2 + 5.0
    mask = torch.ones(4, 32)

    grad_norms = {}
    for b in (1.0, 10.0):
        scaled = 5.0 + b * (base - 5.0)
        batch = ma.MomentBatch(
            x_enc=scaled.unsqueeze(1),
            input_mask=mask,
            dataset=np.array(["x"] * 4),
            domain=np.array(["x"] * 4),
            frequency=np.array(["H"] * 4),
            scale=torch.full((4,), b),
            batch_seed=123,
        )
        model.zero_grad(set_to_none=True)
        result = ma.forward(model, batch, "moment_original")
        result.per_example_loss_masked.mean().backward()
        grad_norms[b] = (
            sum(
                float(p.grad.norm() ** 2)
                for p in model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

    ratio = grad_norms[10.0] / grad_norms[1.0]
    assert abs(ratio - 100.0) / 100.0 < 0.02


def test_normalized_space_removes_scale_dependence():
    """Acceptance criterion: normalized-space runs must not inherit the
    original-space b**2 gradient scaling -- RevIN normalizes the scale away
    before the loss is computed, so the gradient ratio should be ~1, not
    ~100, for the same b=1 vs b=10 pair used in the original-space test."""
    model = ma.build_moment_model(_tiny_model_config(), seed=0)

    torch.manual_seed(0)
    base = torch.randn(4, 32) * 2 + 5.0
    mask = torch.ones(4, 32)

    grad_norms = {}
    for b in (1.0, 10.0):
        scaled = 5.0 + b * (base - 5.0)
        batch = ma.MomentBatch(
            x_enc=scaled.unsqueeze(1),
            input_mask=mask,
            dataset=np.array(["x"] * 4),
            domain=np.array(["x"] * 4),
            frequency=np.array(["H"] * 4),
            scale=torch.full((4,), b),
            batch_seed=123,
        )
        model.zero_grad(set_to_none=True)
        result = ma.forward(model, batch, "moment_normalized")
        result.per_example_loss_masked.mean().backward()
        grad_norms[b] = (
            sum(
                float(p.grad.norm() ** 2)
                for p in model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

    ratio = grad_norms[10.0] / grad_norms[1.0]
    assert abs(ratio - 1.0) < 0.01


def test_training_step_safely_clips_large_original_space_gradient():
    model = ma.build_moment_model(_tiny_model_config(), seed=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    values = torch.linspace(-1.0, 1.0, 32) * 1e15
    batch = ma.MomentBatch(
        x_enc=values.repeat(4, 1).unsqueeze(1),
        input_mask=torch.ones(4, 32),
        dataset=np.array(["x"] * 4),
        domain=np.array(["x"] * 4),
        frequency=np.array(["H"] * 4),
        scale=torch.full((4,), 1e15),
        batch_seed=123,
    )

    metrics = ma.training_step_metrics(
        model, batch, "moment_original", optimizer, grad_clip_norm=1.0
    )

    assert not metrics["step_skipped"]
    assert np.isfinite(metrics["grad_norm_before_clip"])
    assert np.isclose(metrics["grad_norm_after_clip"], 1.0)
    assert metrics["clipped"]


def test_checkpoint_roundtrip(tiny_corpus, tmp_path):
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = ma.make_batch(index, rows, cache)

    model = ma.build_moment_model(_tiny_model_config(), seed=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    ma.training_step_metrics(
        model, batch, "moment_original", optimizer, grad_clip_norm=1.0
    )

    from src.training import tsfm as train_mod

    path = tmp_path / "ckpt.pt"
    train_mod.save_checkpoint(path, model, optimizer, step=1)

    model2 = ma.build_moment_model(_tiny_model_config(), seed=1)  # different init
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-3)
    step = train_mod.load_checkpoint(path, model2, optimizer2)
    assert step == 1
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_scheme_reproduces_the_vendored_revin():
    """`PopulationStdScheme` must equal MOMENT's own RevIN exactly.

    If it drifts, disabling the backbone's normalizer silently changes
    training numerics, which is the class of bug the module exists to prevent.
    """
    from src.models.vendor.moment.models.layers.revin import RevIN

    torch.manual_seed(0)
    context = torch.randn(8, 128, dtype=torch.float32) * 100.0 + 5.0
    valid = torch.ones(8, 128, dtype=torch.float32)
    valid[0, :40] = 0.0
    valid[3, 100:] = 0.0

    revin = RevIN(num_features=1, eps=1e-5, affine=False)
    normalized = revin(x=context.unsqueeze(1), mask=valid, mode="norm")

    scheme = ma.NORMALIZATION.scheme
    loc, scale = scheme.statistics(context, valid)
    assert torch.allclose(loc, revin.mean.squeeze(), atol=0, rtol=0)
    assert torch.allclose(scale, revin.stdev.squeeze(), atol=0, rtol=0)

    stats = norm.TransformStats(loc, scale, scale <= 1e-5, scheme)
    assert torch.equal(stats.forward(context), normalized.squeeze(1))


def test_disabled_backbone_reproduces_the_original_bit_identically(tiny_corpus):
    """MOMENT normalizes over visible positions only, so the mask comes first.

    `vendor/moment/models/moment.py:303` derives the RevIN statistics from
    `mask * input_mask`, the randomly drawn pretrain mask intersected with the
    input mask, which keeps the positions being reconstructed out of them.
    Hoisting normalization out of this backbone therefore means hoisting the
    mask draw with it and passing the mask back in.
    """
    from src.models.vendor.moment.models.layers.revin import RevIN

    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = ma.make_batch(index, rows, cache)

    # build_moment_model disables the backbone's normalizer, so the
    # pre-refactor reference is reconstructed by putting a real RevIN back.
    reference = ma.build_moment_model(_tiny_model_config(), seed=0)
    reference.normalizer = RevIN(num_features=1, eps=1e-5, affine=False)
    # train() mode so dropout draws too: the mask must come off the same
    # stream position it would have inside the forward pass.
    reference.train()
    with torch.random.fork_rng():
        torch.manual_seed(batch.batch_seed)
        with torch.no_grad():
            out_reference = reference(x_enc=batch.x_enc, input_mask=batch.input_mask)

    external = ma.build_moment_model(_tiny_model_config(), seed=0)
    external.train()
    module = ma.NORMALIZATION.module("moment_normalized")
    with torch.random.fork_rng():
        torch.manual_seed(batch.batch_seed)
        drawn_mask = external.mask_generator.generate_mask(
            x=batch.x_enc, input_mask=batch.input_mask
        )
        z_context, stats = module.transform_input(
            batch.x_enc.squeeze(1), drawn_mask * batch.input_mask
        )
        with torch.no_grad():
            out_external = external(
                x_enc=z_context.unsqueeze(1),
                input_mask=batch.input_mask,
                mask=drawn_mask,
            )

    # Drawing the mask outside the forward pass must not move the RNG stream.
    assert torch.equal(drawn_mask, out_reference.pretrain_mask)
    assert torch.equal(
        out_reference.metadata["normalized_reconstruction"],
        out_external.metadata["normalized_reconstruction"],
    )
    assert torch.equal(
        stats.inverse(out_external.metadata["normalized_reconstruction"]),
        out_reference.reconstruction,
    )


def test_identity_revin_is_the_identity():
    x = torch.randn(4, 1, 32) * 50.0
    revin = ma.IdentityRevIN()

    assert torch.equal(revin(x, mode="norm"), x)
    assert torch.equal(revin(x, mode="denorm"), x)

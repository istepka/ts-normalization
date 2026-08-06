import numpy as np
import torch

from src.tsfm_pretraining import moment_adapter as ma
from src.tsfm_pretraining import window_index as wi


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
    """The normalized_mse the adapter reports must equal manually normalizing
    the model's own reconstruction/target with its own RevIN statistics --
    i.e. moment_normalized is genuinely computed pre-inverse-transform, not a
    rescaled copy of the original-space loss."""
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=1)
    batch = ma.make_batch(index, rows, cache)
    model = ma.build_moment_model(_tiny_model_config(), seed=0)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(batch.batch_seed)
        out = model(x_enc=batch.x_enc, input_mask=batch.input_mask)

    mean, stdev = out.metadata["revin_mean"], out.metadata["revin_stdev"]
    manual_normalized_target = (batch.x_enc - mean) / stdev
    manual_diff = (
        out.metadata["normalized_reconstruction"] - manual_normalized_target
    ) ** 2
    train_mask = (1.0 - out.pretrain_mask) * batch.input_mask
    manual_mse = (manual_diff.squeeze(1) * train_mask).sum(dim=1) / train_mask.sum(
        dim=1
    ).clamp_min(1.0)

    result = ma.forward(model, batch, "moment_normalized")
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
            frequency=np.array(["x"] * 4),
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
            frequency=np.array(["x"] * 4),
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


def test_checkpoint_roundtrip(tiny_corpus, tmp_path):
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = ma.make_batch(index, rows, cache)

    model = ma.build_moment_model(_tiny_model_config(), seed=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    ma.training_step_metrics(
        model, batch, "moment_original", optimizer, grad_clip_norm=1.0
    )

    from src.tsfm_pretraining import train as train_mod

    path = tmp_path / "ckpt.pt"
    train_mod.save_checkpoint(path, model, optimizer, step=1)

    model2 = ma.build_moment_model(_tiny_model_config(), seed=1)  # different init
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-3)
    step = train_mod.load_checkpoint(path, model2, optimizer2)
    assert step == 1
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)

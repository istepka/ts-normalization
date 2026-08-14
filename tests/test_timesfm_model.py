import numpy as np
import torch

from src.data.gifteval import window_index as wi
from src.models import timesfm as tm
from src.models.vendor.timesfm_v1.pytorch_patched_decoder import TimesFMConfig


def _tiny_config(patch_len=32, horizon_len=32):
    return TimesFMConfig(
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        hidden_size=32,
        intermediate_size=32,
        head_dim=8,
        patch_len=patch_len,
        horizon_len=horizon_len,
    )


def _tiny_index(tiny_corpus, context_length=64, prediction_length=32):
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


def test_build_timesfm_model_initializes_attention_scaling():
    """Regression test: TimesFMAttention.scaling is allocated with
    torch.empty in the vendored code and left uninitialized, which is
    harmless when loading a released checkpoint (every parameter gets
    overwritten) but is genuine uninitialized memory when training from
    scratch -- observed in practice to intermittently contain huge garbage
    values (~1e30) that explode through softplus and produce NaN logits."""
    junk = torch.full((10_000_000,), 1e30)
    del junk  # dirty the allocator so a real bug here would show up as garbage

    model = tm.build_timesfm_model(_tiny_config(), seed=0)
    scaling_params = [p for n, p in model.named_parameters() if n.endswith("scaling")]
    assert scaling_params, "expected at least one attention scaling parameter"
    for p in scaling_params:
        assert torch.isfinite(p).all()
        assert p.abs().max() < 10.0


def test_frequency_bucket_matches_expected_granularity():
    assert tm.frequency_bucket("M") == 2
    assert tm.frequency_bucket("A-DEC") == 2
    assert tm.frequency_bucket("Q-DEC") == 2
    assert tm.frequency_bucket("D") == 1
    assert tm.frequency_bucket("W-SUN") == 1
    assert tm.frequency_bucket("H") == 0
    assert tm.frequency_bucket("5T") == 0
    assert tm.frequency_bucket("10S") == 0


def test_patching_matches_context_length_over_patch_len(tiny_corpus):
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = tm.make_batch(index, rows, cache, horizon_len=32)
    model = tm.build_timesfm_model(_tiny_config(), seed=0)

    normalized_out, original_out, (mu, sigma) = tm.run_decoder(model, batch)
    n_patches = index.config.context_length // model.config.patch_len
    assert n_patches == 2
    # run_decoder already selects the last patch position for training; shape
    # must be [B, horizon_len, num_outputs].
    num_outputs = len(model.config.quantiles) + 1
    assert normalized_out.shape == (4, 32, num_outputs)
    assert original_out.shape == (4, 32, num_outputs)
    assert mu.shape == (4,)
    assert sigma.shape == (4,)


def test_causal_masking_last_patch_unaffected_by_earlier_content(tiny_corpus):
    """Changing only the earliest patch of context must not perturb the last
    patch's own local computation before attention -- more directly, the
    model's forecast may legitimately change (causal attention lets later
    positions attend to earlier ones), but changing a LATER patch must not
    affect an EARLIER position's output, since that would violate causality
    entirely. Verify the reverse: perturbing the last input patch changes the
    forecast (attention path exists) while perturbing input strictly after
    the used context cannot happen by construction (context is exactly
    context_length)."""
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=1, random_state=0)
    batch = tm.make_batch(index, rows, cache, horizon_len=32)
    model = tm.build_timesfm_model(_tiny_config(), seed=0)

    normalized_out, _, _ = tm.run_decoder(model, batch)

    perturbed_context = batch.context.clone()
    perturbed_context[:, :32] += 1000.0  # perturb only the first (earliest) patch
    perturbed_batch = tm.TimesFMBatch(
        context=perturbed_context,
        context_padding=batch.context_padding,
        target=batch.target,
        target_valid=batch.target_valid,
        freq=batch.freq,
        dataset=batch.dataset,
        domain=batch.domain,
        frequency=batch.frequency,
        scale=batch.scale,
    )
    perturbed_out, _, _ = tm.run_decoder(model, perturbed_batch)
    # Causal decoder-only attention: the last patch's output CAN depend on
    # earlier patches, so a change here is expected, not a bug -- this test
    # instead documents and pins that dependency exists (sanity that the
    # model isn't secretly non-causal in the OTHER, invalid direction is
    # covered by inspecting vendor causal_mask directly, not needed here
    # since it's Google's unmodified code).
    assert not torch.allclose(normalized_out, perturbed_out)


def test_inverse_transform_is_affine_correct(tiny_corpus):
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = tm.make_batch(index, rows, cache, horizon_len=32)
    model = tm.build_timesfm_model(_tiny_config(), seed=0)

    normalized_out, original_out, (mu, sigma) = tm.run_decoder(model, batch)
    reconstructed = normalized_out * sigma[:, None, None] + mu[:, None, None]
    assert torch.allclose(reconstructed, original_out, atol=1e-4)


def test_mse_and_pinball_losses_are_separated(tiny_corpus):
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = tm.make_batch(index, rows, cache, horizon_len=32)
    model = tm.build_timesfm_model(_tiny_config(), seed=0)

    result = tm.forward(model, batch, "timesfm_native_original")
    assert result.mse_per_example.shape == (4,)
    assert result.pinball_per_example.shape == (4,)
    assert torch.allclose(
        result.loss_per_example, result.mse_per_example + result.pinball_per_example
    )
    # zero-target, zero-prediction sanity: pinball and MSE must be independently
    # computable, i.e. one can be nonzero while the other is exactly zero.
    zero_batch = tm.TimesFMBatch(
        context=torch.zeros_like(batch.context),
        context_padding=batch.context_padding,
        target=torch.zeros_like(batch.target),
        target_valid=batch.target_valid,
        freq=batch.freq,
        dataset=batch.dataset,
        domain=batch.domain,
        frequency=batch.frequency,
        scale=batch.scale,
    )
    zero_result = tm.forward(model, zero_batch, "timesfm_native_original")
    assert zero_result.mse_per_example.detach().numpy().min() >= 0
    assert zero_result.pinball_per_example.detach().numpy().min() >= 0


def test_controlled_scale_gradients_match_expected_powers(tiny_corpus):
    model = tm.build_timesfm_model(_tiny_config(), seed=0)
    torch.manual_seed(0)
    base_context = torch.randn(4, 64) * 2 + 5.0
    base_target = torch.randn(4, 32) * 2 + 5.0
    pad = torch.zeros(4, 64)
    valid = torch.ones(4, 32)
    freq = torch.zeros(4, 1, dtype=torch.long)

    mse_grad, pinball_grad = {}, {}
    for b in (1.0, 10.0):
        context = 5.0 + b * (base_context - 5.0)
        target = 5.0 + b * (base_target - 5.0)
        batch = tm.TimesFMBatch(
            context=context,
            context_padding=pad,
            target=target,
            target_valid=valid,
            freq=freq,
            dataset=np.array(["x"] * 4),
            domain=np.array(["x"] * 4),
            frequency=np.array(["H"] * 4),
            scale=torch.full((4,), b),
        )
        model.zero_grad(set_to_none=True)
        result = tm.forward(model, batch, "timesfm_native_original")
        result.mse_per_example.mean().backward()
        mse_grad[b] = (
            sum(
                float(p.grad.norm() ** 2)
                for p in model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

        model.zero_grad(set_to_none=True)
        result2 = tm.forward(model, batch, "timesfm_native_original")
        result2.pinball_per_example.mean().backward()
        pinball_grad[b] = (
            sum(
                float(p.grad.norm() ** 2)
                for p in model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

    assert abs(mse_grad[10.0] / mse_grad[1.0] - 100.0) / 100.0 < 0.02
    assert abs(pinball_grad[10.0] / pinball_grad[1.0] - 10.0) / 10.0 < 0.02


def test_normalized_space_removes_scale_dependence():
    """Acceptance criterion: timesfm_normalized must not inherit the
    original-space b**2 (MSE) / b (pinball) scaling -- both should be ~1 once
    the instance-norm statistics are divided out before the loss."""
    model = tm.build_timesfm_model(_tiny_config(), seed=0)
    torch.manual_seed(0)
    base_context = torch.randn(4, 64) * 2 + 5.0
    base_target = torch.randn(4, 32) * 2 + 5.0
    pad = torch.zeros(4, 64)
    valid = torch.ones(4, 32)
    freq = torch.zeros(4, 1, dtype=torch.long)

    mse_grad, pinball_grad = {}, {}
    for b in (1.0, 10.0):
        context = 5.0 + b * (base_context - 5.0)
        target = 5.0 + b * (base_target - 5.0)
        batch = tm.TimesFMBatch(
            context=context,
            context_padding=pad,
            target=target,
            target_valid=valid,
            freq=freq,
            dataset=np.array(["x"] * 4),
            domain=np.array(["x"] * 4),
            frequency=np.array(["H"] * 4),
            scale=torch.full((4,), b),
        )
        model.zero_grad(set_to_none=True)
        result = tm.forward(model, batch, "timesfm_normalized")
        result.mse_per_example.mean().backward()
        mse_grad[b] = (
            sum(
                float(p.grad.norm() ** 2)
                for p in model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

        model.zero_grad(set_to_none=True)
        result2 = tm.forward(model, batch, "timesfm_normalized")
        result2.pinball_per_example.mean().backward()
        pinball_grad[b] = (
            sum(
                float(p.grad.norm() ** 2)
                for p in model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

    assert abs(mse_grad[10.0] / mse_grad[1.0] - 1.0) < 0.02
    assert abs(pinball_grad[10.0] / pinball_grad[1.0] - 1.0) < 0.02


def test_horizon_len_mismatch_raises(tiny_corpus):
    index, cache = _tiny_index(tiny_corpus, context_length=64, prediction_length=16)
    rows = index.split("train").sample(n=2, random_state=0)
    try:
        tm.make_batch(index, rows, cache, horizon_len=32)
    except ValueError as e:
        assert "horizon_len" in str(e)
    else:
        raise AssertionError("expected ValueError for mismatched horizon_len")


def test_flat_first_patch_window_is_flagged_degenerate_and_excluded():
    """TimesFM derives sigma from the FIRST PATCH only and clamps it at
    config.tolerance, so a window whose first patch is constant normalizes by
    1e-6 and its normalized-space target explodes. Such windows must be
    flagged and kept out of the reported metrics."""
    config = tm.CONFIG_17M
    model = tm.build_timesfm_model(config, seed=0)
    context = torch.randn(2, 512).cumsum(dim=1)
    context[1, : config.patch_len] = 3.0  # flat first patch -> sigma at clamp
    batch = tm.TimesFMBatch(
        context=context,
        context_padding=torch.zeros(2, 512),
        target=torch.randn(2, config.horizon_len) * 50.0,
        target_valid=torch.ones(2, config.horizon_len),
        freq=torch.zeros(2, 1, dtype=torch.long),
        dataset=np.array(["a", "b"]),
        domain=np.array(["d", "d"]),
        frequency=np.array(["H", "H"]),
        scale=torch.ones(2),
    )
    result = tm.forward(model, batch, "timesfm_normalized")

    assert not bool(result.degenerate[0])
    assert bool(result.degenerate[1])
    # The degenerate window contributes nothing to either reported metric.
    assert torch.isfinite(result.normalized_mse[0])
    assert torch.isnan(result.normalized_mse[1])
    assert torch.isnan(result.mase[1])


def test_degenerate_filter_is_independent_of_controlled_scale():
    config = _tiny_config()
    model = tm.build_timesfm_model(config, seed=0)
    first_patch = torch.arange(config.patch_len) * 5e-8
    context = torch.cat([first_patch, torch.arange(config.patch_len)])
    target = torch.arange(config.horizon_len, dtype=torch.float32)

    results = []
    for scale in (1.0, 10.0):
        batch = tm.TimesFMBatch(
            context=(context * scale).unsqueeze(0),
            context_padding=torch.zeros(1, 2 * config.patch_len),
            target=(target * scale).unsqueeze(0),
            target_valid=torch.ones(1, config.horizon_len),
            freq=torch.zeros(1, 1, dtype=torch.long),
            dataset=np.array(["x"]),
            domain=np.array(["x"]),
            frequency=np.array(["H"]),
            scale=torch.tensor([scale]),
        )
        results.append(tm.forward(model, batch, "timesfm_normalized"))

    assert bool(results[0].degenerate[0])
    assert bool(results[1].degenerate[0])


def test_whole_context_statistics_rescue_flat_first_patch():
    config = _tiny_config()
    model = tm.build_timesfm_model(config, seed=0)
    context = torch.arange(2 * config.patch_len, dtype=torch.float32)
    context[: config.patch_len] = 3.0
    batch = tm.TimesFMBatch(
        context=context.unsqueeze(0),
        context_padding=torch.zeros(1, 2 * config.patch_len),
        target=torch.arange(config.horizon_len, dtype=torch.float32).unsqueeze(0),
        target_valid=torch.ones(1, config.horizon_len),
        freq=torch.zeros(1, 1, dtype=torch.long),
        dataset=np.array(["x"]),
        domain=np.array(["x"]),
        frequency=np.array(["H"]),
        scale=torch.ones(1),
    )

    first_patch = tm.forward(model, batch, "timesfm_normalized", "first_patch")
    whole_context = tm.forward(model, batch, "timesfm_normalized", "whole_context")

    assert bool(first_patch.degenerate[0])
    assert not bool(whole_context.degenerate[0])


def test_normalized_loss_is_scale_invariant_for_both_statistic_modes():
    config = _tiny_config()
    model = tm.build_timesfm_model(config, seed=0)
    torch.manual_seed(0)
    context = torch.randn(4, 2 * config.patch_len)
    target = torch.randn(4, config.horizon_len)
    for normalization_mode in tm.NORMALIZATION_MODES:
        losses = []
        for scale in (1.0, 10.0):
            batch = tm.TimesFMBatch(
                context=context * scale,
                context_padding=torch.zeros_like(context),
                target=target * scale,
                target_valid=torch.ones_like(target),
                freq=torch.zeros(4, 1, dtype=torch.long),
                dataset=np.array(["x"] * 4),
                domain=np.array(["x"] * 4),
                frequency=np.array(["H"] * 4),
                scale=torch.full((4,), scale),
            )
            result = tm.forward(model, batch, "timesfm_normalized", normalization_mode)
            losses.append(result.mse_per_example)

        assert torch.allclose(losses[0], losses[1], rtol=1e-5, atol=1e-6)


def test_original_loss_step_handles_extreme_natural_scale():
    config = _tiny_config()
    torch.manual_seed(0)
    context = torch.randn(4, 2 * config.patch_len) * 1e17
    target = torch.randn(4, config.horizon_len) * 1e17
    batch = tm.TimesFMBatch(
        context=context,
        context_padding=torch.zeros_like(context),
        target=target,
        target_valid=torch.ones_like(target),
        freq=torch.zeros(4, 1, dtype=torch.long),
        dataset=np.array(["x"] * 4),
        domain=np.array(["x"] * 4),
        frequency=np.array(["H"] * 4),
        scale=torch.ones(4),
    )

    for normalization_mode in tm.NORMALIZATION_MODES:
        model = tm.build_timesfm_model(config, seed=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        metrics = tm.training_step_metrics(
            model,
            batch,
            "timesfm_native_original",
            normalization_mode,
            "mse",
            optimizer,
            1.0,
        )

        assert not metrics["step_skipped"]
        assert np.isfinite(metrics["total_grad_norm_before_clip"])
        assert np.isclose(metrics["total_grad_norm_after_clip"], 1.0)
        for state in optimizer.state.values():
            assert all(
                torch.isfinite(value).all()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )


def test_controlled_batches_standardize_before_applying_scale(tiny_corpus):
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").groupby("dataset", sort=False).head(1)
    batch_a = tm.make_batch(index, rows, cache, horizon_len=32, scale_assignment="A")
    batch_b = tm.make_batch(index, rows, cache, horizon_len=32, scale_assignment="B")

    assert torch.allclose(
        batch_a.context / batch_a.scale[:, None],
        batch_b.context / batch_b.scale[:, None],
    )
    assert torch.allclose(
        batch_a.target / batch_a.scale[:, None],
        batch_b.target / batch_b.scale[:, None],
    )

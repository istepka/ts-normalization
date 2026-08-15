"""Every scheme must reproduce the backbone normalizer it replaces.

These are the tests that make the normalization module safe to adopt. If a
scheme drifts from its upstream reference, swapping a backbone's internal
normalization for this module silently changes training numerics, which is
exactly the class of bug this refactor exists to prevent.
"""

import pytest
import torch

from src.models import normalization as norm


@pytest.fixture
def window():
    torch.manual_seed(0)
    context = torch.randn(8, 128, dtype=torch.float32) * 100.0 + 5.0
    valid = torch.ones(8, 128, dtype=torch.float32)
    valid[0, :40] = 0.0
    valid[3, 100:] = 0.0
    return context, valid


def test_standard_scheme_matches_patch_transformer(window):
    context, _ = window
    valid = torch.ones_like(context)
    scheme = norm.StandardScheme(eps=1e-5)
    loc, scale = scheme.statistics(context, valid)

    expected_loc = context.mean(dim=1)
    expected_scale = context.std(dim=1) + 1e-5

    assert torch.allclose(loc, expected_loc, atol=1e-5)
    assert torch.allclose(scale, expected_scale, atol=1e-4)


def test_moment_revin_scheme_matches_vendored_revin(window):
    context, valid = window
    from src.models.vendor.moment.models.layers.revin import RevIN

    revin = RevIN(num_features=1, eps=1e-5, affine=False)
    x = context.unsqueeze(1)
    normalized = revin(x=x, mask=valid, mode="norm")

    scheme = norm.MomentRevINScheme(eps=1e-5)
    loc, scale = scheme.statistics(context, valid)

    assert torch.allclose(loc, revin.mean.squeeze(), atol=0, rtol=0)
    assert torch.allclose(scale, revin.stdev.squeeze(), atol=0, rtol=0)

    stats = norm.TransformStats(loc, scale, scale <= 1e-5, scheme)
    assert torch.equal(stats.forward(context), normalized.squeeze(1))


def test_chronos2_scheme_matches_instance_norm(window):
    context, valid = window
    from chronos.chronos_bolt import InstanceNorm

    masked = torch.where(valid.bool(), context, torch.nan)
    for use_arcsinh in (True, False):
        instance_norm = InstanceNorm(eps=1e-5, use_arcsinh=use_arcsinh)
        expected, (loc, scale) = instance_norm(masked)

        scheme = norm.Chronos2Scheme(eps=1e-5, use_arcsinh=use_arcsinh)
        stats_loc, stats_scale = scheme.statistics(context, valid)
        assert torch.equal(stats_loc, loc.squeeze(-1))
        assert torch.equal(stats_scale, scale.squeeze(-1))

        stats = norm.TransformStats(stats_loc, stats_scale, stats_scale <= 1e-5, scheme)
        # equal_nan because masked positions stay NaN through the transform.
        torch.testing.assert_close(
            stats.forward(masked), expected, rtol=0, atol=0, equal_nan=True
        )
        # arcsinh is not affine but it is invertible.
        round_tripped = stats.inverse(stats.forward(context))
        assert torch.allclose(round_tripped, context, rtol=1e-4, atol=1e-2)


def test_moirai2_std_scheme_matches_packed_std_scaler(window):
    context, valid = window
    from src.models.vendor.moirai2.packed_scaler import PackedStdScaler

    batch, length = context.shape
    patch_size = 16
    patched = context.view(batch, -1, patch_size)
    observed = valid.view(batch, -1, patch_size).bool()
    num_patches = length // patch_size
    sample_id = torch.ones(batch, num_patches, dtype=torch.long)
    variate_id = torch.zeros(batch, num_patches, dtype=torch.long)

    scaler = PackedStdScaler(correction=1, minimum_scale=1e-5)
    loc, scale = scaler(patched, observed, sample_id, variate_id)

    scheme = norm.Moirai2StdScheme(correction=1, minimum_scale=1e-5)
    stats_loc, stats_scale = scheme.statistics(context, valid)

    assert torch.allclose(stats_loc, loc[:, 0, 0], atol=0, rtol=0)
    assert torch.allclose(stats_scale, scale[:, 0, 0], atol=0, rtol=0)


def test_moirai2_absmean_scheme_matches_packed_abs_mean_scaler(window):
    context, valid = window
    from src.models.vendor.moirai2.packed_scaler import PackedAbsMeanScaler

    batch, length = context.shape
    patch_size = 16
    patched = context.view(batch, -1, patch_size)
    observed = valid.view(batch, -1, patch_size).bool()
    num_patches = length // patch_size
    sample_id = torch.ones(batch, num_patches, dtype=torch.long)
    variate_id = torch.zeros(batch, num_patches, dtype=torch.long)

    scaler = PackedAbsMeanScaler()
    loc, scale = scaler(patched, observed, sample_id, variate_id)

    scheme = norm.Moirai2AbsMeanScheme()
    stats_loc, stats_scale = scheme.statistics(context, valid)

    assert torch.allclose(stats_loc, loc[:, 0, 0], atol=0, rtol=0)
    assert torch.allclose(stats_scale, scale[:, 0, 0], atol=0, rtol=0)


def test_timesfm_first_patch_scheme_matches_forward_transform(window):
    context, valid = window
    from src.models.timesfm import CONFIG_17M, build_timesfm_model

    model = build_timesfm_model(CONFIG_17M, seed=0)
    padding = 1.0 - valid
    _, _, stats_reference, _ = model._preprocess_input(
        input_ts=context, input_padding=padding
    )
    mu, sigma = stats_reference

    scheme = norm.TimesFMScheme(
        mode="first_patch",
        patch_len=CONFIG_17M.patch_len,
        tolerance=CONFIG_17M.tolerance,
        pad_val=CONFIG_17M.pad_val,
    )
    loc, scale = scheme.statistics(context, valid)

    assert torch.equal(loc, mu)
    assert torch.equal(scale, sigma)


def test_timesfm_whole_context_scheme_matches_repo_preprocessing(window):
    context, valid = window
    import numpy as np

    from src.models.timesfm import CONFIG_17M, TimesFMBatch, build_timesfm_model

    model = build_timesfm_model(CONFIG_17M, seed=0)
    batch = TimesFMBatch(
        context=context,
        context_padding=1.0 - valid,
        target=torch.zeros(context.shape[0], CONFIG_17M.horizon_len),
        target_valid=torch.ones(context.shape[0], CONFIG_17M.horizon_len),
        freq=torch.zeros(context.shape[0], 1, dtype=torch.long),
        dataset=np.array(["d"] * context.shape[0]),
        domain=np.array(["x"] * context.shape[0]),
        frequency=np.array(["H"] * context.shape[0]),
        scale=torch.ones(context.shape[0]),
    )
    from src.models.timesfm import _preprocess_whole_context

    _, _, (mu, sigma) = _preprocess_whole_context(model, batch)

    scheme = norm.TimesFMScheme(
        mode="whole_context",
        patch_len=CONFIG_17M.patch_len,
        tolerance=CONFIG_17M.tolerance,
        pad_val=CONFIG_17M.pad_val,
    )
    loc, scale = scheme.statistics(context, valid)

    assert torch.equal(loc, mu)
    assert torch.equal(scale, sigma)


def test_sit_and_revin_align_into_opposite_spaces(window):
    context, valid = window
    scheme = norm.StandardScheme()
    target = torch.randn(context.shape[0], 16) * 100.0

    sit = norm.SIT(scheme)
    _, stats = sit.transform_input(context, valid)
    output = torch.randn_like(target)

    sit_output, sit_target = sit.transform_target_and_output(output, target, stats)
    assert torch.equal(sit_output, output)
    assert torch.equal(sit_target, stats.forward(target))

    revin = norm.RevIN(scheme)
    revin_output, revin_target = revin.transform_target_and_output(
        output, target, stats
    )
    assert torch.equal(revin_target, target)
    assert torch.equal(revin_output, stats.inverse(output))

    # The two arms are the same forward pass read in two spaces.
    assert torch.allclose(stats.inverse(sit_target), target, atol=1e-2)


def test_align_target_output_is_an_alias(window):
    context, valid = window
    sit = norm.SIT(norm.StandardScheme())
    _, stats = sit.transform_input(context, valid)
    output = torch.randn(context.shape[0], 16)
    target = torch.randn(context.shape[0], 16)

    assert torch.equal(
        sit.align_target_output(output, target, stats)[1],
        sit.transform_target_and_output(output, target, stats)[1],
    )


def test_quantile_predictions_broadcast_against_per_window_statistics(window):
    context, valid = window
    scheme = norm.StandardScheme()
    module = norm.RevIN(scheme)
    _, stats = module.transform_input(context, valid)

    quantile_output = torch.randn(context.shape[0], 9, 16)
    denormalized = stats.inverse(quantile_output)

    assert denormalized.shape == quantile_output.shape
    expected = quantile_output * stats.scale.view(-1, 1, 1) + stats.loc.view(-1, 1, 1)
    assert torch.equal(denormalized, expected)


def test_causal_norm_uses_the_prior_window_only(window):
    context, valid = window
    extra_context = torch.randn_like(context) * 3.0 + 50.0
    scheme = norm.StandardScheme()

    causal = norm.SIT(scheme, apply_causal_norm=True)
    _, causal_stats = causal.transform_input(
        context, valid, extra_context=extra_context
    )
    expected_loc, expected_scale = scheme.statistics(
        extra_context, torch.ones_like(extra_context)
    )

    assert torch.equal(causal_stats.loc, expected_loc)
    assert torch.equal(causal_stats.scale, expected_scale)

    # Without the flag the extra context is ignored entirely.
    non_causal = norm.SIT(scheme, apply_causal_norm=False)
    _, stats = non_causal.transform_input(context, valid, extra_context=extra_context)
    assert torch.equal(stats.loc, scheme.statistics(context, valid)[0])


def test_causal_norm_without_extra_context_fails_loudly(window):
    context, valid = window
    module = norm.SIT(norm.StandardScheme(), apply_causal_norm=True)
    with pytest.raises(ValueError, match="apply_causal_norm"):
        module.transform_input(context, valid)


def test_degenerate_windows_are_flagged(window):
    context, valid = window
    context[2] = 7.0  # constant window, no usable normalized-space target
    module = norm.SIT(norm.StandardScheme(eps=1e-5))
    _, stats = module.transform_input(context, valid)

    assert bool(stats.degenerate[2])
    assert not bool(stats.degenerate[1])


def test_build_scheme_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown scheme"):
        norm.build_scheme("not_a_scheme")

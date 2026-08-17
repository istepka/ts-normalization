"""The backbone-agnostic half of the normalization module.

Each scheme is pinned against the upstream implementation it reproduces in
that model's own adapter test, since the reference lives there. What is tested
here is the machinery every model shares: statistics threading, the SIT/RevIN
contrast, causal statistics, and the `BackboneNormalization` prototype a new
baseline subclasses.
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


def test_floored_std_scheme_floors_under_the_square_root(window):
    """The floor is added to the variance, so the smallest scale is its root."""
    context, _ = window
    context = torch.zeros_like(context)
    scheme = norm.FlooredStdScheme(correction=1, minimum_scale=1e-4)
    _, scale = scheme.statistics(context, torch.ones_like(context))

    assert torch.allclose(scale, torch.full_like(scale, 1e-2))
    assert scheme.scale_floor == pytest.approx(1e-2)


def test_abs_mean_scheme_has_zero_location(window):
    context, valid = window
    scheme = norm.AbsMeanScheme()
    loc, scale = scheme.statistics(context, valid)

    assert torch.equal(loc, torch.zeros_like(loc))
    assert torch.allclose(scale, (context * valid).abs().sum(1) / valid.sum(1))


def test_arcsinh_scheme_is_invertible_but_not_affine(window):
    context, valid = window
    scheme = norm.ArcsinhStdScheme(eps=1e-5, use_arcsinh=True)
    module = norm.SIT(scheme)
    _, stats = module.transform_input(context, valid)

    round_tripped = stats.inverse(stats.forward(context))
    assert torch.allclose(round_tripped, context, rtol=1e-4, atol=1e-2)

    # Affine would make the difference of two forwards independent of offset.
    a, b = context[:, :1], context[:, 1:2]
    spread = stats.forward(a) - stats.forward(b)
    shifted = stats.forward(a + 50.0) - stats.forward(b + 50.0)
    assert not torch.allclose(spread, shifted, rtol=1e-3, atol=1e-3)


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

    # The two conditions are the same forward pass read in two spaces.
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


class _ToyNormalization(norm.BackboneNormalization):
    """The whole contract a new baseline has to satisfy."""

    normalized_condition = "toy_normalized"
    original_condition = "toy_original"

    def __init__(self):
        super().__init__(norm.StandardScheme())
        self.disabled = False

    def disable(self, model) -> None:
        self.disabled = True


def test_backbone_normalization_picks_the_module_for_each_condition():
    backbone = _ToyNormalization()

    assert isinstance(backbone.module("toy_normalized"), norm.SIT)
    assert isinstance(backbone.module("toy_original"), norm.RevIN)
    assert backbone.module("toy_normalized").scheme is backbone.scheme


def test_backbone_normalization_conditions_need_no_instance():
    """The training loop validates cfg.condition before it has a model config."""
    assert _ToyNormalization.conditions() == ("toy_normalized", "toy_original")


def test_backbone_normalization_rejects_an_unknown_condition():
    with pytest.raises(ValueError, match="unknown condition"):
        _ToyNormalization().module("toy_sideways")

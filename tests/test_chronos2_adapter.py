import numpy as np
import torch

from src.data.gifteval import window_index as wi
from src.models import chronos2 as ca
from src.models import normalization as norm


def _tiny_config() -> ca.Chronos2Config:
    return ca.Chronos2Config(
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


def _tiny_index(tiny_corpus):
    root, domain_map = tiny_corpus
    config = wi.WindowIndexConfig(
        context_length=64,
        prediction_length=32,
        stride=80,
        val_series_fraction=0.25,
        base_seed=0,
    )
    index = wi.build_window_index(root, ["synth_a", "synth_b"], domain_map, config)
    return index, wi.SeriesCache(root)


def _with_native_normalization(model, config):
    """Puts the backbone's own InstanceNorm back.

    `build_chronos2_model` disables it, so a test that wants the pre-refactor
    behavior as a reference has to restore it. InstanceNorm holds no
    parameters, so this does not perturb the model's weights.
    """
    from chronos.chronos_bolt import InstanceNorm

    model.instance_norm = InstanceNorm(eps=1e-5, use_arcsinh=config.use_arcsinh)
    return model


def test_forward_matches_official_native_loss(tiny_corpus):
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = ca.make_batch(index, rows, cache)
    config = _tiny_config()
    model = ca.build_chronos2_model(config, seed=0)

    result = ca.forward(model, batch, "chronos2_normalized")
    # The official path normalizes internally, so it needs its InstanceNorm.
    official = _with_native_normalization(model, config)(
        context=batch.context,
        context_mask=batch.context_valid,
        future_target=batch.target,
        future_target_mask=batch.target_valid,
        num_output_patches=2,
    )

    assert result.loss_per_example.shape == (4,)
    assert torch.allclose(result.loss_per_example.mean(), official.loss, atol=1e-6)
    assert result.original_point_forecast.shape == (4, 32)


def test_original_predictions_are_inverse_of_normalized_predictions(tiny_corpus):
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=1)
    batch = ca.make_batch(index, rows, cache)
    model = ca.build_chronos2_model(_tiny_config(), seed=0)

    normalized, original, stats = ca.run_model(model, batch)

    assert torch.allclose(stats.inverse(normalized), original, atol=1e-5)


def test_controlled_scale_changes_only_original_loss_gradient():
    model = ca.build_chronos2_model(_tiny_config(), seed=0)
    torch.manual_seed(0)
    base_context = torch.randn(4, 64) * 2 + 5.0
    base_target = torch.randn(4, 32) * 2 + 5.0
    valid_context = torch.ones_like(base_context)
    valid_target = torch.ones_like(base_target)

    gradients = {condition: {} for condition in ca.CONDITIONS}
    for condition in ca.CONDITIONS:
        for b in (1.0, 10.0):
            batch = ca.Chronos2Batch(
                context=5.0 + b * (base_context - 5.0),
                context_valid=valid_context,
                target=5.0 + b * (base_target - 5.0),
                target_valid=valid_target,
                dataset=np.array(["x"] * 4),
                domain=np.array(["x"] * 4),
                frequency=np.array(["H"] * 4),
                scale=torch.full((4,), b),
            )
            model.zero_grad(set_to_none=True)
            ca.forward(model, batch, condition).loss_per_example.mean().backward()
            gradients[condition][b] = (
                sum(
                    float(parameter.grad.norm() ** 2)
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
                ** 0.5
            )

    normalized_ratio = (
        gradients["chronos2_normalized"][10.0] / gradients["chronos2_normalized"][1.0]
    )
    original_ratio = (
        gradients["chronos2_original"][10.0] / gradients["chronos2_original"][1.0]
    )
    assert abs(normalized_ratio - 1.0) < 0.02
    assert abs(original_ratio - 10.0) / 10.0 < 0.02


def test_training_step_updates_model(tiny_corpus):
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = ca.make_batch(index, rows, cache)
    model = ca.build_chronos2_model(_tiny_config(), seed=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    before = model.output_patch_embedding.output_layer.weight.detach().clone()

    metrics = ca.training_step_metrics(
        model,
        batch,
        "chronos2_normalized",
        optimizer,
        grad_clip_norm=1.0,
    )

    after = model.output_patch_embedding.output_layer.weight.detach()
    assert metrics["loss"] > 0
    assert not torch.equal(before, after)


def test_scheme_reproduces_the_official_instance_norm():
    """`ArcsinhStdScheme` must equal Chronos-2's own InstanceNorm exactly."""
    from chronos.chronos_bolt import InstanceNorm

    torch.manual_seed(0)
    context = torch.randn(8, 128, dtype=torch.float32) * 100.0 + 5.0
    valid = torch.ones(8, 128, dtype=torch.float32)
    valid[0, :40] = 0.0
    valid[3, 100:] = 0.0
    masked = torch.where(valid.bool(), context, torch.nan)

    for use_arcsinh in (True, False):
        instance_norm = InstanceNorm(eps=1e-5, use_arcsinh=use_arcsinh)
        expected, (loc, scale) = instance_norm(masked)

        scheme = norm.ArcsinhStdScheme(eps=1e-5, use_arcsinh=use_arcsinh)
        stats_loc, stats_scale = scheme.statistics(context, valid)
        assert torch.equal(stats_loc, loc.squeeze(-1))
        assert torch.equal(stats_scale, scale.squeeze(-1))

        stats = norm.TransformStats(stats_loc, stats_scale, stats_scale <= 1e-5, scheme)
        # equal_nan because masked positions stay NaN through the transform.
        torch.testing.assert_close(
            stats.forward(masked), expected, rtol=0, atol=0, equal_nan=True
        )


def test_disabled_backbone_reproduces_the_original_bit_identically(tiny_corpus):
    """Normalizing outside the backbone must move no training numerics."""
    config = _tiny_config()
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = ca.make_batch(index, rows, cache)

    # build_chronos2_model disables the normalizer, so the pre-refactor
    # reference is reconstructed by putting the native InstanceNorm back and
    # driving the backbone's own encode path.
    reference = _with_native_normalization(
        ca.build_chronos2_model(config, seed=0), config
    )
    reference.eval()
    num_output_patches = (
        batch.target.shape[1] // reference.chronos_config.output_patch_size
    )
    with torch.no_grad():
        encoder_output, loc_scale, _, _ = reference.encode(
            context=batch.context,
            context_mask=batch.context_valid,
            num_output_patches=num_output_patches,
        )
        hidden = encoder_output.last_hidden_state[:, -num_output_patches:]
        projected = reference.output_patch_embedding(hidden)
        batch_size = projected.shape[0]
        normalized_reference = (
            projected.view(
                batch_size,
                num_output_patches,
                reference.num_quantiles,
                reference.chronos_config.output_patch_size,
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, reference.num_quantiles, -1)
        )
        original_reference = reference.instance_norm.inverse(
            normalized_reference.reshape(batch_size, -1), loc_scale
        ).reshape_as(normalized_reference)

    external = ca.build_chronos2_model(config, seed=0)
    backbone = ca.Chronos2Normalization.from_model(external)
    external.eval()

    # run_model normalizes internally now, so it takes the raw batch.
    with torch.no_grad():
        normalized_external, original_external, _ = ca.run_model(external, batch)

    assert backbone.scheme.use_arcsinh == config.use_arcsinh
    assert torch.equal(normalized_reference, normalized_external)
    # RevIN reads the same forward pass in the original space.
    assert torch.equal(original_external, original_reference)


def test_identity_instance_norm_is_the_identity():
    x = torch.randn(4, 1, 32) * 50.0
    instance_norm = ca.IdentityInstanceNorm()

    normalized, (loc, scale) = instance_norm(x)
    assert torch.equal(normalized, x)
    assert torch.equal(loc, torch.zeros_like(loc))
    assert torch.equal(scale, torch.ones_like(scale))
    assert torch.equal(instance_norm.inverse(x, (loc, scale)), x)


def test_mixed_geometry_batch_scores_each_window_independently(mixed_length_corpus):
    """The whole justification for padding to the fixed maximum instead of to
    the batch maximum: a window's loss must not depend on which other windows
    happen to share its batch, or two conditions drawing the same schedule
    would still see different per-example losses."""
    root, domain_map = mixed_length_corpus
    config = wi.WindowIndexConfig(
        context_length=64,
        prediction_length=32,
        stride=96,
        val_series_fraction=0.25,
        base_seed=0,
        min_context_length=16,
        min_prediction_length=8,
    )
    index = wi.build_window_index(root, ["mixed"], domain_map, config)
    cache = wi.SeriesCache(root)
    rows = index.table
    assert rows["context_length"].nunique() > 1

    model_config = _tiny_config()
    model = ca.build_chronos2_model(model_config, seed=0)
    model.eval()

    with torch.no_grad():
        together = ca.forward(
            model, ca.make_batch(index, rows, cache), "chronos2_normalized"
        ).loss_per_example
        alone = torch.stack(
            [
                ca.forward(
                    model,
                    ca.make_batch(index, rows.iloc[[i]], cache),
                    "chronos2_normalized",
                ).loss_per_example[0]
                for i in range(len(rows))
            ]
        )
    assert torch.allclose(together, alone, atol=1e-5)

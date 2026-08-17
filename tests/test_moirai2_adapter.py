import numpy as np
import torch

from src.data.gifteval import window_index as wi
from src.models import moirai2 as ma
from src.models import normalization as norm


def _tiny_config() -> ma.Moirai2Config:
    return ma.Moirai2Config(
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


def test_run_model_matches_manual_reshape_of_raw_model_output(tiny_corpus):
    config = _tiny_config()
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = ma.make_batch(index, rows, cache, config)
    model = ma.build_moirai2_model(config, seed=0)

    normalized_preds, normalized_target, stats = ma.run_model(model, batch, config)
    # The backbone's scaler is disabled, so the raw call takes the sequence
    # already normalized, which is what run_model feeds it.
    z_sequence = stats.forward(batch.target.reshape(4, -1)).view_as(batch.target)
    raw_preds, raw_scaled_target = model(
        z_sequence,
        batch.observed_mask,
        batch.sample_id,
        batch.time_id,
        batch.variate_id,
        batch.prediction_mask,
        training_mode=True,
    )

    pred_position = config.context_token_length - 1
    manual = (
        raw_preds[:, pred_position]
        .view(4, model.num_predict_token, model.num_quantiles, model.patch_size)
        .permute(0, 2, 1, 3)
        .reshape(4, model.num_quantiles, config.predict_horizon)
    )
    manual_target = raw_scaled_target[:, config.context_token_length :].reshape(
        4, config.predict_horizon
    )

    assert normalized_preds.shape == (4, 3, 32)
    assert normalized_target.shape == (4, 32)
    assert stats.loc.shape == (4,)
    assert stats.scale.shape == (4,)
    assert torch.equal(normalized_preds, manual)
    assert torch.equal(normalized_target, manual_target)


def test_original_predictions_are_inverse_of_normalized_predictions(tiny_corpus):
    config = _tiny_config()
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=1)
    batch = ma.make_batch(index, rows, cache, config)
    model = ma.build_moirai2_model(config, seed=0)

    normalized_preds, _, stats = ma.run_model(model, batch, config)
    result = ma.forward(model, batch, "moirai2_original", config)

    reconstructed_median = stats.inverse(normalized_preds[:, 1])
    assert torch.allclose(
        reconstructed_median, result.original_point_forecast, atol=1e-5
    )


def test_evaluation_disables_attention_dropout(tiny_corpus):
    config = ma.Moirai2Config(
        **{
            **_tiny_config().__dict__,
            "attn_dropout_p": 0.2,
            "dropout_p": 0.2,
        }
    )
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=2)
    batch = ma.make_batch(index, rows, cache, config)
    model = ma.build_moirai2_model(config, seed=0).eval()

    with torch.no_grad():
        first = ma.forward(model, batch, "moirai2_normalized", config)
        second = ma.forward(model, batch, "moirai2_normalized", config)

    assert torch.equal(first.original_point_forecast, second.original_point_forecast)
    assert torch.equal(first.normalized_mse, second.normalized_mse)
    assert torch.equal(first.mase, second.mase)


def test_controlled_scale_changes_only_original_loss_gradient():
    config = _tiny_config()
    model = ma.build_moirai2_model(config, seed=0)
    num_patches = config.num_patches

    torch.manual_seed(0)
    base = torch.randn(4, num_patches, config.patch_size) * 2 + 5.0
    observed_mask = torch.ones(4, num_patches, config.patch_size, dtype=torch.bool)
    sample_id = torch.ones(4, num_patches, dtype=torch.long)
    time_id = torch.arange(num_patches, dtype=torch.long).unsqueeze(0).expand(4, -1)
    variate_id = torch.zeros(4, num_patches, dtype=torch.long)
    prediction_mask = torch.zeros(4, num_patches, dtype=torch.bool)
    prediction_mask[:, config.context_token_length :] = True

    gradients = {condition: {} for condition in ma.CONDITIONS}
    for condition in ma.CONDITIONS:
        for b in (1.0, 10.0):
            batch = ma.Moirai2Batch(
                target=5.0 + b * (base - 5.0),
                observed_mask=observed_mask,
                sample_id=sample_id,
                time_id=time_id.clone(),
                variate_id=variate_id,
                prediction_mask=prediction_mask,
                dataset=np.array(["x"] * 4),
                domain=np.array(["x"] * 4),
                frequency=np.array(["H"] * 4),
                scale=torch.full((4,), b),
            )
            model.zero_grad(set_to_none=True)
            ma.forward(
                model, batch, condition, config
            ).loss_per_example.mean().backward()
            gradients[condition][b] = (
                sum(
                    float(parameter.grad.norm() ** 2)
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
                ** 0.5
            )

    normalized_ratio = (
        gradients["moirai2_normalized"][10.0] / gradients["moirai2_normalized"][1.0]
    )
    original_ratio = (
        gradients["moirai2_original"][10.0] / gradients["moirai2_original"][1.0]
    )
    assert abs(normalized_ratio - 1.0) < 0.05
    assert abs(original_ratio - 10.0) / 10.0 < 0.05


def test_training_step_updates_model(tiny_corpus):
    config = _tiny_config()
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = ma.make_batch(index, rows, cache, config)
    model = ma.build_moirai2_model(config, seed=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    before = model.out_proj.output_layer.weight.detach().clone()

    metrics = ma.training_step_metrics(
        model,
        batch,
        "moirai2_normalized",
        config,
        optimizer,
        grad_clip_norm=1.0,
    )

    after = model.out_proj.output_layer.weight.detach()
    assert metrics["loss"] > 0
    assert not torch.equal(before, after)
    assert isinstance(metrics["dataset"], np.ndarray)


def _patched(context, valid, patch_size=16):
    batch, length = context.shape
    num_patches = length // patch_size
    return (
        context.view(batch, -1, patch_size),
        valid.view(batch, -1, patch_size).bool(),
        torch.ones(batch, num_patches, dtype=torch.long),
        torch.zeros(batch, num_patches, dtype=torch.long),
    )


def test_scheme_reproduces_the_packed_std_scaler():
    """`FlooredStdScheme` must equal uni2ts's PackedStdScaler exactly."""
    from src.models.vendor.moirai2.packed_scaler import PackedStdScaler

    torch.manual_seed(0)
    context = torch.randn(8, 128, dtype=torch.float32) * 100.0 + 5.0
    valid = torch.ones(8, 128, dtype=torch.float32)
    valid[0, :40] = 0.0
    valid[3, 100:] = 0.0

    loc, scale = PackedStdScaler(correction=1, minimum_scale=1e-5)(
        *_patched(context, valid)
    )
    scheme = norm.FlooredStdScheme(correction=1, minimum_scale=1e-5)
    stats_loc, stats_scale = scheme.statistics(context, valid)

    assert torch.allclose(stats_loc, loc[:, 0, 0], atol=0, rtol=0)
    assert torch.allclose(stats_scale, scale[:, 0, 0], atol=0, rtol=0)


def test_abs_mean_scheme_reproduces_the_packed_abs_mean_scaler():
    from src.models.vendor.moirai2.packed_scaler import PackedAbsMeanScaler

    torch.manual_seed(0)
    context = torch.randn(8, 128, dtype=torch.float32) * 100.0 + 5.0
    valid = torch.ones(8, 128, dtype=torch.float32)
    valid[0, :40] = 0.0

    loc, scale = PackedAbsMeanScaler()(*_patched(context, valid))
    stats_loc, stats_scale = norm.AbsMeanScheme().statistics(context, valid)

    assert torch.allclose(stats_loc, loc[:, 0, 0], atol=0, rtol=0)
    assert torch.allclose(stats_scale, scale[:, 0, 0], atol=0, rtol=0)


def test_disabled_backbone_reproduces_the_original_bit_identically(tiny_corpus):
    """Normalizing outside the backbone must move no training numerics."""
    config = _tiny_config()
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = ma.make_batch(index, rows, cache, config)

    # build_moirai2_model disables the scaler, so the pre-refactor reference
    # is reconstructed by putting the native PackedStdScaler back and driving
    # the backbone's own normalizing path.
    from src.models.vendor.moirai2.packed_scaler import PackedStdScaler

    reference = ma.build_moirai2_model(config, seed=0)
    reference.scaler = PackedStdScaler()
    reference.eval()
    batch_size = batch.target.shape[0]
    context_mask = batch.observed_mask * ~batch.prediction_mask.unsqueeze(-1)
    with torch.no_grad():
        loc, scale = reference.scaler(
            batch.target, context_mask, batch.sample_id, batch.variate_id
        )
        preds_reference, scaled_target_reference = reference(
            batch.target,
            batch.observed_mask,
            batch.sample_id,
            batch.time_id,
            batch.variate_id,
            batch.prediction_mask,
            training_mode=True,
        )

    external = ma.build_moirai2_model(config, seed=0)
    external.eval()
    with torch.no_grad():
        preds_external, target_external, stats = ma.run_model(external, batch, config)

    assert torch.equal(stats.loc, loc[:, 0, 0])
    assert torch.equal(stats.scale, scale[:, 0, 0])

    pred_position = config.context_token_length - 1
    expected_preds = (
        preds_reference[:, pred_position]
        .view(
            batch_size,
            reference.num_predict_token,
            reference.num_quantiles,
            reference.patch_size,
        )
        .permute(0, 2, 1, 3)
        .reshape(batch_size, reference.num_quantiles, config.predict_horizon)
    )
    expected_target = scaled_target_reference[:, config.context_token_length :].reshape(
        batch_size, config.predict_horizon
    )

    assert torch.equal(expected_preds, preds_external)
    assert torch.equal(expected_target, target_external)


def test_disable_keeps_the_degenerate_floor():
    """The adapter derives its degenerate floor from the scaler's own value."""
    config = _tiny_config()
    model = ma.build_moirai2_model(config, seed=0)
    minimum_scale = model.scaler.minimum_scale
    ma.Moirai2Normalization(minimum_scale=minimum_scale).disable(model)

    assert model.scaler.minimum_scale == minimum_scale
    target = torch.randn(2, 6, 16)
    loc, scale = model.scaler(
        target,
        torch.ones_like(target, dtype=torch.bool),
        torch.ones(2, 6, dtype=torch.long),
        torch.zeros(2, 6, dtype=torch.long),
    )
    assert torch.equal(loc, torch.zeros_like(loc))
    assert torch.equal(scale, torch.ones_like(scale))


def test_stats_target_matches_the_backbone_scaled_target(tiny_corpus):
    """`forward` takes the SIT target from `stats`, not from `scaled_target`.

    The backbone still returns its own normalized target, so the two must
    agree or the normalized condition would silently score against a
    different target than the one the model was trained on.
    """
    config = _tiny_config()
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = ma.make_batch(index, rows, cache, config)
    model = ma.build_moirai2_model(config, seed=0)

    _, normalized_target, stats = ma.run_model(model, batch, config)
    context_patches = config.context_token_length
    future_target = batch.target[:, context_patches:].reshape(4, config.predict_horizon)

    assert torch.equal(stats.forward(future_target), normalized_target)

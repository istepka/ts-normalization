import numpy as np
import torch

from src.tsfm_pretraining import moirai2_adapter as ma
from src.tsfm_pretraining import window_index as wi


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

    normalized_preds, normalized_target, loc, scale = ma.run_model(model, batch, config)
    raw_preds, raw_scaled_target = model(
        batch.target,
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
    assert loc.shape == (4,)
    assert scale.shape == (4,)
    assert torch.equal(normalized_preds, manual)
    assert torch.equal(normalized_target, manual_target)


def test_original_predictions_are_inverse_of_normalized_predictions(tiny_corpus):
    config = _tiny_config()
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=1)
    batch = ma.make_batch(index, rows, cache, config)
    model = ma.build_moirai2_model(config, seed=0)

    normalized_preds, _, loc, scale = ma.run_model(model, batch, config)
    result = ma.forward(model, batch, "moirai2_original", config)

    reconstructed_median = normalized_preds[:, 1] * scale.unsqueeze(-1) + (
        loc.unsqueeze(-1)
    )
    assert torch.allclose(
        reconstructed_median, result.original_point_forecast, atol=1e-5
    )


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

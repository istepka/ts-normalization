import numpy as np
import torch

from src.data.gifteval import window_index as wi
from src.models import chronos2 as ca


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


def test_forward_matches_official_native_loss(tiny_corpus):
    index, cache = _tiny_index(tiny_corpus)
    rows = index.split("train").sample(n=4, random_state=0)
    batch = ca.make_batch(index, rows, cache)
    model = ca.build_chronos2_model(_tiny_config(), seed=0)

    result = ca.forward(model, batch, "chronos2_normalized")
    official = model(
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
    reconstructed = model.instance_norm.inverse(
        normalized.reshape(normalized.shape[0], -1), stats
    ).reshape_as(normalized)

    assert torch.allclose(reconstructed, original, atol=1e-5)


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

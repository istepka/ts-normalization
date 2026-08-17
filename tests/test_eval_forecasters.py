"""Each model's build_forecaster against the eval protocol.

Tiny configs and freshly saved checkpoints: what is under test is that the
adapters satisfy the protocol and return original-space quantiles in the
right shape, not that an untrained model forecasts well.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.eval import predict
from src.eval.protocol import Forecaster, build_forecaster
from src.eval.suites import EvalSeries

CONTEXT = 64


def _cfg(model: str) -> OmegaConf:
    return OmegaConf.create(
        {
            "model": model,
            "seed": 0,
            "window_index": {"context_length": CONTEXT, "prediction_length": 128},
            "timesfm": {
                "config_size": "17m",
                "normalization_mode": "first_patch",
                "objective": "mse",
                "grad_clip_norm": 1.0,
            },
            "chronos2": {
                "context_length": CONTEXT,
                "prediction_length": 32,
                "patch_size": 16,
                "d_model": 32,
                "d_kv": 8,
                "d_ff": 64,
                "num_layers": 2,
                "num_heads": 4,
                "dropout_rate": 0.0,
                "initializer_factor": 0.05,
                "quantiles": [0.1, 0.5, 0.9],
                "use_arcsinh": True,
                "grad_clip_norm": 1.0,
            },
            "moirai2": {
                "context_length": CONTEXT,
                "predict_horizon": 32,
                "patch_size": 16,
                "d_model": 64,
                "d_ff": 64,
                "num_layers": 2,
                "max_seq_len": CONTEXT,
                "attn_dropout_p": 0.0,
                "dropout_p": 0.0,
                "scaling": True,
                "quantile_levels": [0.1, 0.5, 0.9],
                "grad_clip_norm": 1.0,
            },
        }
    )


def _checkpoint(cfg, tmp_path):
    """Builds the model the same way training does and saves a checkpoint in
    training's format, so build_forecaster is exercised against a real one."""
    from src.models import chronos2, moirai2, timesfm
    from src.training.tsfm import TIMESFM_CONFIGS

    if cfg.model == "timesfm":
        model = timesfm.build_timesfm_model(
            TIMESFM_CONFIGS[cfg.timesfm.config_size], seed=0
        )
    elif cfg.model == "chronos2":
        model = chronos2.build_chronos2_model(
            chronos2.Chronos2Config(
                **OmegaConf.to_container(cfg.chronos2, resolve=True)
            ),
            seed=0,
        )
    else:
        model = moirai2.build_moirai2_model(
            moirai2.Moirai2Config(**OmegaConf.to_container(cfg.moirai2, resolve=True)),
            seed=0,
        )
    path = tmp_path / "checkpoint_step1.pt"
    torch.save({"model": model.state_dict(), "optimizer": {}, "step": 1}, path)
    return path


@pytest.mark.parametrize("model", ["timesfm", "chronos2", "moirai2"])
def test_build_forecaster_satisfies_the_protocol(model, tmp_path):
    cfg = _cfg(model)
    forecaster = build_forecaster(cfg, _checkpoint(cfg, tmp_path), "cpu")

    assert isinstance(forecaster, Forecaster)
    assert forecaster.context_length == CONTEXT
    assert 0.5 in forecaster.quantiles
    assert forecaster.horizon > 0


@pytest.mark.parametrize("model", ["timesfm", "chronos2", "moirai2"])
def test_forecasters_return_original_space_quantiles(model, tmp_path):
    """Shape [N, horizon, Q] with the quantile axis last, and finite values
    on the series' own scale. Chronos-2 and Moirai both emit [B, Q, H]
    internally, so a missing transpose here would silently mismatch the
    quantile and horizon axes whenever Q and H happen to be equal."""
    cfg = _cfg(model)
    forecaster = build_forecaster(cfg, _checkpoint(cfg, tmp_path), "cpu")

    rng = np.random.default_rng(0)
    context = rng.normal(loc=50.0, scale=5.0, size=(3, CONTEXT))
    valid = np.ones((3, CONTEXT))
    out = forecaster.predict(context, valid, ["D"] * 3)

    assert out.shape == (3, forecaster.horizon, len(forecaster.quantiles))
    assert np.isfinite(out).all()


@pytest.mark.parametrize("model", ["timesfm", "chronos2", "moirai2"])
def test_ragged_suites_survive_the_full_predict_path(model, tmp_path):
    """A suite of series far shorter than the context, which is the normal
    case for M1/M3/Tourism, must come back with one forecast per series."""
    cfg = _cfg(model)
    forecaster = build_forecaster(cfg, _checkpoint(cfg, tmp_path), "cpu")

    series = [
        EvalSeries(
            suite="test",
            subset="s",
            item_id=f"i{i}",
            history=np.arange(1.0, 9.0 + i),
            actual=np.arange(6.0),
            period=1,
            freq="Y",
        )
        for i in range(5)
    ]
    out = predict.run(forecaster, series, batch_size=2)

    assert out.values.shape == (5, 6, len(forecaster.quantiles))
    assert np.isfinite(out.values).all()

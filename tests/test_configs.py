import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from omegaconf.errors import (
    ConfigKeyError,
    MissingMandatoryValue,
    ValidationError,
)

from src.configs import ToyConfig, TsfmConfig, validate_config


def test_runtime_configs_compose_and_validate():
    with initialize(version_base=None, config_path="../conf"):
        toy = compose(config_name="config")
        scale_swap = compose(config_name="scale_swap")
        tsfm = [
            compose(config_name=name)
            for name in (
                "tsfm_chronos2",
                "tsfm_moirai2",
                "tsfm_moment",
                "tsfm_timesfm",
            )
        ]

    validate_config(toy, ToyConfig)
    validate_config(scale_swap, ToyConfig)
    for cfg in tsfm:
        validate_config(cfg, TsfmConfig)

    assert "modes" not in toy
    assert "seed" not in toy
    assert scale_swap.data.kind == "real_scale_swap"
    assert {cfg.window_index.prediction_length for cfg in tsfm} == {128}
    assert {cfg.wandb.project for cfg in tsfm} == {"tsfm_pretraining"}


def test_structured_config_rejects_unknown_fields():
    cfg = OmegaConf.create({"seeds": [0], "unexpected": True})

    with pytest.raises(ConfigKeyError):
        validate_config(cfg, ToyConfig)


def test_structured_config_rejects_wrong_types():
    cfg = OmegaConf.create({"train": {"steps": "not-an-int"}})

    with pytest.raises(ValidationError):
        validate_config(cfg, TsfmConfig)


def test_structured_config_rejects_missing_values():
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="tsfm_timesfm")
    del cfg.wandb.project

    with pytest.raises(MissingMandatoryValue):
        validate_config(cfg, TsfmConfig)

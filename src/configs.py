"""Shared Hydra config validation plumbing.

The experiment-specific structured schemas live with the experiment they
belong to: `src.loss_space.configs.ToyConfig` and
`src.tsfm_pretraining.configs.TsfmConfig`. This module holds only what both
of them need: the `WandbConfig` shape they both embed, and the
`validate_config` dispatcher.
"""

from dataclasses import dataclass

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class WandbConfig:
    entity: str = MISSING
    project: str = MISSING
    mode: str = MISSING
    experiment: str = MISSING


def validate_config(cfg: DictConfig, schema: type) -> DictConfig:
    """Validate a composed Hydra config against a structured schema."""
    from src.loss_space.configs import ToyConfig
    from src.tsfm_pretraining.configs import TsfmConfig

    validated = OmegaConf.merge(OmegaConf.structured(schema), cfg)
    if schema is ToyConfig:
        OmegaConf.to_container(validated, resolve=False, throw_on_missing=True)
    elif schema is TsfmConfig:
        common_fields = (
            "model",
            "device",
            "condition",
            "experiment_kind",
            "scale_assignment",
            "scale_b_low",
            "scale_b_high",
            "seed",
            "output_dir",
            "corpus",
            "window_index",
            "dataset_weights",
            "train",
            "wandb",
        )
        scalar_fields = {
            "model",
            "device",
            "condition",
            "experiment_kind",
            "scale_assignment",
            "scale_b_low",
            "scale_b_high",
            "seed",
            "output_dir",
            "dataset_weights",
        }
        for field_name in common_fields:
            if field_name in scalar_fields:
                if OmegaConf.is_missing(validated, field_name):
                    validated[field_name]
            else:
                OmegaConf.to_container(
                    validated[field_name], resolve=False, throw_on_missing=True
                )
        OmegaConf.to_container(
            validated[cfg.model], resolve=False, throw_on_missing=True
        )
    else:
        raise TypeError(f"unsupported config schema {schema!r}")
    return cfg

"""Loss-space comparison toy: normalized vs. original-space training loss.

Runs the core comparison plus two controls, all from a single Hydra config, and
repeats every setup over several seeds so plots can show mean +/- 1 SE:

- normalized / original (base, heterogeneous variance): the core comparison.
- original_equalvar: equal-variance control; per-category disparity must vanish.
- original_gradmatch: gradient-norm-matched control; disparity must persist, proving
  the effect is loss-space bias rather than a global learning-rate artifact.

Each run reports to Weights & Biases (grouped per setup, tagged with experiment /
setup / loss mode / seed) and contributes to the saved figures/metrics.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src.data import SyntheticTSDataset
from src.plots import (
    plot_forecast_evolution,
    plot_global_nmse,
    plot_grad_magnitude,
    plot_nmse_panels,
    plot_qualitative,
)
from src.train import Trainer

SETUP_LABELS = ["normalized", "original", "original_equalvar", "original_gradmatch"]


def build_run_specs(cfg: DictConfig, seed: int) -> list[tuple[str, DictConfig, str]]:
    """Four setups at a fixed seed. normalized/original share the same seeded
    dataset and init so they differ only in loss space; the controls reuse the
    same seed."""

    def variant(**overrides):
        c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        c.seed = seed
        for key, value in overrides.items():
            OmegaConf.update(c, key, value)
        return c

    base = variant()
    equalvar = variant(**{"data.equal_variance": True})
    gradmatch = variant(**{"train.grad_norm_match": True})
    return [
        ("normalized", base, "normalized"),
        ("original", base, "original"),
        ("original_equalvar", equalvar, "original"),
        ("original_gradmatch", gradmatch, "original"),
    ]


def verification_summary(results: dict, names: list[str]) -> dict:
    """Final nMSE (per-category and global) and near-init gradient ratios,
    averaged across seeds, for the falsifiable checks."""

    summary = {}
    for label in results:
        histories = results[label]["histories"]
        fn = np.array([np.stack(h["nmse"], axis=0)[-1] for h in histories]).mean(axis=0)
        final_global = np.array([h["global_nmse"][-1] for h in histories])
        summary[label] = {
            "final_nmse": {n: float(v) for n, v in zip(names, fn)},
            "nmse_spread_ratio": float(fn.max() / fn.min()),
            "final_global_nmse": float(final_global.mean()),
        }
    # Step-0 gradient ratio: the b^2 = sigma^2 scaling is exact at init.
    grad = np.array([h["grad_mag"][0] for h in results["original"]["histories"]]).mean(
        axis=0
    )
    summary["original"]["init_grad_ratio_to_smallest"] = {
        n: float(v / grad.min()) for n, v in zip(names, grad)
    }
    return summary


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        label: {"histories": [], "model": None, "dataset": None}
        for label in SETUP_LABELS
    }
    for seed in cfg.seeds:
        for label, run_cfg, mode in build_run_specs(cfg, seed):
            generator = torch.Generator().manual_seed(run_cfg.seed)
            dataset = SyntheticTSDataset(run_cfg, generator)
            wandb_run = wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=f"{label}-seed{seed}",
                group=f"{cfg.wandb.experiment}/{label}",
                job_type=mode,
                tags=[cfg.wandb.experiment, label, mode, f"seed{seed}"],
                mode=cfg.wandb.mode,
                config=OmegaConf.to_container(run_cfg, resolve=True),
                reinit=True,
            )
            trainer = Trainer(run_cfg, dataset, mode, wandb_run)
            history = trainer.run()
            wandb_run.finish()
            results[label]["histories"].append(history)
            results[label]["model"] = trainer.model  # last seed, for qualitative
            results[label]["dataset"] = dataset

    names = results["normalized"]["dataset"].category_names

    # Write metrics before plotting so a plotting hiccup never wastes a full run.
    summary = verification_summary(results, names)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    band = cfg.plot.band
    plot_nmse_panels(
        results,
        names,
        labels=["normalized", "original"],
        titles=["Normalized-space loss", "Original-space loss"],
        band=band,
        out_path=out_dir / "nmse_core.png",
    )
    plot_nmse_panels(
        results,
        names,
        labels=["original_equalvar", "original_gradmatch"],
        titles=["Original + equal variance", "Original + grad-norm matched"],
        band=band,
        out_path=out_dir / "nmse_controls.png",
    )
    plot_global_nmse(results, SETUP_LABELS, band, out_dir / "nmse_global.png")
    plot_grad_magnitude(results, names, band, out_dir / "grad_magnitude.png")
    plot_qualitative(
        cfg,
        results["original"]["dataset"],
        results["original"]["model"],
        out_dir / "forecasts_original.png",
    )
    plot_qualitative(
        cfg,
        results["normalized"]["dataset"],
        results["normalized"]["model"],
        out_dir / "forecasts_normalized.png",
    )
    # Forecast evolution (one representative seed) — the disparate convergence rate.
    plot_forecast_evolution(
        results["original"]["histories"][-1],
        names,
        "Forecast evolution — original-space loss",
        out_dir / "forecast_evolution_original.png",
    )
    plot_forecast_evolution(
        results["normalized"]["histories"][-1],
        names,
        "Forecast evolution — normalized-space loss",
        out_dir / "forecast_evolution_normalized.png",
    )


if __name__ == "__main__":
    main()

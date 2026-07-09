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

from src.captions import write_captions
from src.data import SyntheticTSDataset
from src.plots import (
    gif_forecast_evolution,
    gif_nmse_convergence,
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

    # Write metrics + forecast snapshot data before plotting, so a plotting hiccup
    # never wastes a full run and figures can be restyled later without retraining
    # (see scripts/replot_forecasts.py).
    summary = verification_summary(results, names)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    for label in ("normalized", "original"):
        h = results[label]["histories"][-1]
        np.savez(
            out_dir / f"forecast_data_{label}.npz",
            names=np.array(names),
            steps=np.array(h["forecast_steps"]),
            pred=np.array(h["forecast_pred"]),
            context=h["probe_context"],
            target=h["probe_target"],
        )
    # Per-step metric histories for every setup, stacked over seeds, so the nMSE
    # figures can be restyled/zoomed later without retraining (scripts/replot_metrics.py).
    for label in SETUP_LABELS:
        hs = results[label]["histories"]
        np.savez(
            out_dir / f"metrics_{label}.npz",
            names=np.array(names),
            step=np.array(hs[0]["step"]),
            nmse=np.array([np.stack(h["nmse"], axis=0) for h in hs]),  # [S, E, C]
            global_nmse=np.array([h["global_nmse"] for h in hs]),  # [S, E]
            grad_mag=np.array([np.stack(h["grad_mag"], axis=0) for h in hs]),  # [S,E,C]
        )

    render_figures(cfg, results, names, out_dir)


def render_figures(cfg, results, names, out_dir: Path):
    """Render every figure twice — a screen version (PNG+PDF, titles) into `out_dir`
    and a paper version (PDF only, no titles, minimal margins) into the sibling
    `{out_dir}_paper` directory with per-figure caption files — plus the GIFs."""
    band = cfg.plot.band
    paper_dir = out_dir.parent / f"{out_dir.name}_paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    for target, paper in ((out_dir, False), (paper_dir, True)):
        plot_nmse_panels(
            results,
            names,
            ["normalized", "original"],
            ["Normalized-space loss", "Original-space loss"],
            band,
            target / "nmse_core.png",
            paper,
        )
        plot_nmse_panels(
            results,
            names,
            ["original_equalvar", "original_gradmatch"],
            ["Original + equal variance", "Original + grad-norm matched"],
            band,
            target / "nmse_controls.png",
            paper,
        )
        plot_global_nmse(results, SETUP_LABELS, band, target / "nmse_global.png", paper)
        plot_grad_magnitude(results, names, band, target / "grad_magnitude.png", paper)
        for label in ("original", "normalized"):
            plot_qualitative(
                cfg,
                results[label]["dataset"],
                results[label]["model"],
                target / f"forecasts_{label}.png",
                paper,
            )
            plot_forecast_evolution(
                results[label]["histories"][-1],
                names,
                f"Forecast evolution — {label}-space loss",
                cfg.train.forecast_columns,
                target / f"forecast_evolution_{label}.png",
                paper,
            )
    write_captions(paper_dir)

    # Linear-scale companions (screen dir only), one per zoom window over the first N
    # steps where essentially all convergence happens (the full range crushes it flat).
    for w in cfg.plot.linear_xlim:
        xlim = (0, w)
        plot_nmse_panels(
            results,
            names,
            ["normalized", "original"],
            ["Normalized-space loss", "Original-space loss"],
            band,
            out_dir / f"nmse_core_linear_{w}.png",
            paper=False,
            yscale="linear",
            xlim=xlim,
        )
        plot_nmse_panels(
            results,
            names,
            ["original_equalvar", "original_gradmatch"],
            ["Original + equal variance", "Original + grad-norm matched"],
            band,
            out_dir / f"nmse_controls_linear_{w}.png",
            paper=False,
            yscale="linear",
            xlim=xlim,
        )
        plot_global_nmse(
            results,
            SETUP_LABELS,
            band,
            out_dir / f"nmse_global_linear_{w}.png",
            paper=False,
            yscale="linear",
            xlim=xlim,
        )
    # Linear-scale gradient-magnitude companion (screen dir only).
    plot_grad_magnitude(
        results, names, band, out_dir / "grad_magnitude_linear.png", yscale="linear"
    )

    # Animations for talks / website (screen dir only).
    for label in ("original", "normalized"):
        gif_forecast_evolution(
            results[label]["histories"][-1],
            names,
            out_dir / f"forecast_evolution_{label}.gif",
        )
    gif_nmse_convergence(results, names, band, out_dir / "nmse_convergence.gif")


if __name__ == "__main__":
    main()

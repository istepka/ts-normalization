"""Unified training entrypoint for the GiftEvalPretrain loss-space stages.

Runs exactly one (model, loss-space condition[, scale assignment]) combination
per process invocation and reports to Weights & Biases, matching this
project's existing convention (see src/train.py) of one wandb run per setup,
grouped by experiment tag; sbatch scripts loop over conditions/assignments the
same way scripts/reproducibility/real_scale_swap does for scale-swap A/B.

Both models share: a schedule of window-index positions built once per
(experiment_kind, schedule_seed) so paired conditions -- e.g.
moment_normalized vs moment_original, or scale assignment A vs B -- see
identical windows in identical order (only the loss space or the applied
scale differs); per-(dataset, domain, frequency) windows-processed exposure
counters; and the dispersion/equity metrics from losses.py, logged at
train.eval_every through train.steps.
"""

import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import wandb

from . import gifteval_corpus as gc
from . import losses as L
from . import moment_adapter as ma
from . import timesfm_model as tm
from . import window_index as wi

TIMESFM_CONFIGS = {"17m": tm.CONFIG_17M, "70m": tm.CONFIG_70M}
OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
}


def resolve_window_index(
    cfg: DictConfig, domain_map: dict[str, dict] | None = None
) -> wi.WindowIndex:
    domain_map = domain_map if domain_map is not None else gc.load_domain_map()
    root = Path(cfg.corpus.root)
    dataset_names = (
        list(cfg.corpus.datasets)
        if cfg.corpus.datasets is not None
        else gc.discover_dataset_dirs(root)
    )
    wi_cfg = wi.WindowIndexConfig(
        context_length=cfg.window_index.context_length,
        prediction_length=cfg.window_index.prediction_length,
        stride=cfg.window_index.stride,
        val_series_fraction=cfg.window_index.val_series_fraction,
        min_valid_fraction=cfg.window_index.min_valid_fraction,
        base_seed=cfg.window_index.base_seed,
        max_windows_per_series=cfg.window_index.max_windows_per_series,
    )
    cache_path = cfg.window_index.cache_path
    if cache_path and Path(cache_path).is_file():
        return wi.WindowIndex.load(Path(cache_path), wi_cfg, root)
    index = wi.build_window_index(root, dataset_names, domain_map, wi_cfg)
    if cache_path:
        index.save(Path(cache_path))
    return index


def resolve_dataset_weights(cfg: DictConfig, index: wi.WindowIndex) -> dict[str, float]:
    present = sorted(index.table["dataset"].unique())
    if cfg.dataset_weights is None:
        return {d: 1.0 for d in present}
    weights = OmegaConf.to_container(cfg.dataset_weights, resolve=True)
    missing = set(present) - set(weights)
    if missing:
        raise ValueError(f"dataset_weights missing entries for {sorted(missing)}")
    return weights


def windows_processed_counter() -> dict[str, dict[str, int]]:
    return {"dataset": {}, "domain": {}, "frequency": {}}


def update_windows_processed(counter: dict[str, dict[str, int]], rows) -> None:
    for level, column in (
        ("dataset", "dataset"),
        ("domain", "domain"),
        ("frequency", "frequency"),
    ):
        for value, count in rows[column].value_counts().items():
            counter[level][value] = counter[level].get(value, 0) + int(count)


def dispersion_report(
    per_example_loss: np.ndarray, rows, n_sources_min: int = 2
) -> dict:
    """Gini / unweighted-mean / pooled-mean at dataset, domain, and frequency
    granularity, plus the pooled natural-mixture global error, per the plan's
    "Dispersion and equity metrics" section."""
    report = {"pooled_global_error": L.pooled_mean(per_example_loss)}
    for level in ("dataset", "domain", "frequency"):
        per_source = L.group_mean_by_source(per_example_loss, rows[level].to_numpy())
        metrics = L.dispersion_metrics(per_source)
        if metrics["n_sources"] < n_sources_min:
            continue
        report[level] = {**metrics, "per_source_mean_error": per_source}
    return report


def log_dispersion(wandb_run, report: dict, step: int) -> None:
    flat = {"eval/pooled_global_error": report["pooled_global_error"]}
    for level in ("dataset", "domain", "frequency"):
        if level not in report:
            continue
        flat[f"eval/{level}_gini"] = report[level]["gini"]
        flat[f"eval/{level}_unweighted_mean"] = report[level]["unweighted_mean"]
        flat[f"eval/{level}_n_sources"] = report[level]["n_sources"]
        for name, value in report[level]["per_source_mean_error"].items():
            flat[f"eval/{level}_error/{name}"] = value
    wandb_run.log(flat, step=step)


def sample_eval_rows(index: wi.WindowIndex, n: int, seed: int):
    val = index.split("val")
    if len(val) == 0:
        raise ValueError(
            "val split is empty; lower window_index.val_series_fraction is too small"
        )
    return val.sample(n=min(n, len(val)), random_state=seed)


def run_moment(cfg: DictConfig, index: wi.WindowIndex, wandb_run) -> dict:
    if cfg.condition not in ma.CONDITIONS:
        raise ValueError(f"condition must be one of {ma.CONDITIONS}")
    dataset_weights = resolve_dataset_weights(cfg, index)
    model_cfg = ma.MomentConfig(**OmegaConf.to_container(cfg.moment, resolve=True))
    model = ma.build_moment_model(model_cfg, seed=cfg.seed).to(cfg.device)
    optimizer = OPTIMIZERS[cfg.train.optimizer](model.parameters(), lr=cfg.train.lr)
    cache = wi.SeriesCache(index.corpus_root)

    schedule = wi.build_batch_schedule(
        index,
        "train",
        dataset_weights,
        cfg.train.steps,
        cfg.train.batch_size,
        cfg.train.schedule_seed,
    )
    train_table = index.split("train").reset_index(drop=True)
    scale_assignment = (
        cfg.scale_assignment if cfg.experiment_kind == "controlled_scale" else None
    )

    counter = windows_processed_counter()
    history = {"step": [], "pooled_mse": [], "reports": []}
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, cfg.train.steps + 1):
        rows = train_table.iloc[schedule[step - 1]]
        batch = ma.make_batch(
            index,
            rows,
            cache,
            scale_assignment=scale_assignment,
            b_low=cfg.scale_b_low,
            b_high=cfg.scale_b_high,
        ).to(cfg.device)
        metrics = ma.training_step_metrics(
            model, batch, cfg.condition, optimizer, model_cfg.grad_clip_norm
        )
        update_windows_processed(counter, rows)

        wandb_run.log(
            {
                "train/loss": metrics["loss"],
                "train/grad_norm_before_clip": metrics["grad_norm_before_clip"],
                "train/grad_norm_after_clip": metrics["grad_norm_after_clip"],
                "train/clipped": float(metrics["clipped"]),
                "train/masked_mse": float(
                    np.mean(
                        metrics[
                            "normalized_mse"
                            if cfg.condition == "moment_normalized"
                            else "original_mse"
                        ]
                    )
                ),
                "train/unmasked_mse": float(
                    np.mean(metrics["per_example_loss_unmasked"])
                ),
            },
            step=step,
        )

        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            eval_rows = sample_eval_rows(
                index, cfg.train.eval_batches * cfg.train.batch_size, seed=step
            )
            eval_batch = ma.make_batch(
                index,
                eval_rows,
                cache,
                scale_assignment=scale_assignment,
                b_low=cfg.scale_b_low,
                b_high=cfg.scale_b_high,
            ).to(cfg.device)
            with torch.no_grad():
                eval_out = ma.forward(model, eval_batch, cfg.condition)
            per_example = eval_out.per_example_loss_masked.cpu().numpy()
            report = dispersion_report(per_example, eval_rows)
            log_dispersion(wandb_run, report, step)
            history["step"].append(step)
            history["pooled_mse"].append(report["pooled_global_error"])
            history["reports"].append(report)
            for level in ("dataset", "domain", "frequency"):
                wandb_run.log(
                    {f"exposure/{level}_windows_processed_n": len(counter[level])},
                    step=step,
                )

        if step % cfg.train.checkpoint_every == 0 or step == cfg.train.steps:
            save_checkpoint(
                out_dir / f"checkpoint_step{step}.pt", model, optimizer, step
            )

    summary = finalize_summary(history, counter, cfg)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))
    return summary


def run_timesfm(cfg: DictConfig, index: wi.WindowIndex, wandb_run) -> dict:
    if cfg.condition not in tm.CONDITIONS:
        raise ValueError(f"condition must be one of {tm.CONDITIONS}")
    dataset_weights = resolve_dataset_weights(cfg, index)
    model_config = TIMESFM_CONFIGS[cfg.timesfm.config_size]
    if model_config.horizon_len != cfg.window_index.prediction_length:
        raise ValueError(
            "window_index.prediction_length must equal the TimesFM config's "
            f"horizon_len ({cfg.window_index.prediction_length} != {model_config.horizon_len})"
        )
    model = tm.build_timesfm_model(model_config, seed=cfg.seed).to(cfg.device)
    optimizer = OPTIMIZERS[cfg.train.optimizer](model.parameters(), lr=cfg.train.lr)
    cache = wi.SeriesCache(index.corpus_root)

    schedule = wi.build_batch_schedule(
        index,
        "train",
        dataset_weights,
        cfg.train.steps,
        cfg.train.batch_size,
        cfg.train.schedule_seed,
    )
    train_table = index.split("train").reset_index(drop=True)
    scale_assignment = (
        cfg.scale_assignment if cfg.experiment_kind == "controlled_scale" else None
    )

    counter = windows_processed_counter()
    history = {"step": [], "pooled_mse": [], "reports": []}
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, cfg.train.steps + 1):
        rows = train_table.iloc[schedule[step - 1]]
        batch = tm.make_batch(
            index,
            rows,
            cache,
            model_config.horizon_len,
            scale_assignment=scale_assignment,
            b_low=cfg.scale_b_low,
            b_high=cfg.scale_b_high,
        ).to(cfg.device)
        metrics = tm.training_step_metrics(
            model, batch, cfg.condition, optimizer, cfg.timesfm.grad_clip_norm
        )
        update_windows_processed(counter, rows)

        wandb_run.log(
            {
                "train/loss": metrics["loss"],
                "train/mse": float(np.mean(metrics["mse_per_example"])),
                "train/pinball": float(np.mean(metrics["pinball_per_example"])),
                "train/mse_to_pinball_loss_ratio": metrics["mse_to_pinball_loss_ratio"],
                "train/mse_grad_norm": metrics["mse_grad_norm"],
                "train/total_grad_norm_before_clip": metrics[
                    "total_grad_norm_before_clip"
                ],
                "train/total_grad_norm_after_clip": metrics[
                    "total_grad_norm_after_clip"
                ],
                "train/clipped": float(metrics["clipped"]),
            },
            step=step,
        )

        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            eval_rows = sample_eval_rows(
                index, cfg.train.eval_batches * cfg.train.batch_size, seed=step
            )
            eval_batch = tm.make_batch(
                index,
                eval_rows,
                cache,
                model_config.horizon_len,
                scale_assignment=scale_assignment,
                b_low=cfg.scale_b_low,
                b_high=cfg.scale_b_high,
            ).to(cfg.device)
            with torch.no_grad():
                eval_out = tm.forward(model, eval_batch, cfg.condition)
            per_example = eval_out.mse_per_example.cpu().numpy()
            report = dispersion_report(per_example, eval_rows)
            log_dispersion(wandb_run, report, step)
            history["step"].append(step)
            history["pooled_mse"].append(report["pooled_global_error"])
            history["reports"].append(report)
            for level in ("dataset", "domain", "frequency"):
                wandb_run.log(
                    {f"exposure/{level}_windows_processed_n": len(counter[level])},
                    step=step,
                )

        if step % cfg.train.checkpoint_every == 0 or step == cfg.train.steps:
            save_checkpoint(
                out_dir / f"checkpoint_step{step}.pt", model, optimizer, step
            )

    summary = finalize_summary(history, counter, cfg)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))
    return summary


def finalize_summary(history: dict, counter: dict, cfg: DictConfig) -> dict:
    steps = np.array(history["step"])
    mse = np.array(history["pooled_mse"])
    summary = {
        "windows_processed": counter,
        "final_pooled_mse": float(mse[-1]) if mse.size else None,
        "moment_revision": ma.MOMENT_REVISION if cfg.model == "moment" else None,
        "timesfm_revision": tm.TIMESFM_REVISION if cfg.model == "timesfm" else None,
    }
    if steps.size >= 2:
        cutoff = min(L.TROUGH_STEP_CUTOFF, int(steps.max()))
        summary["log_mse_auc_through_2000"] = L.log_mse_auc(
            steps, mse, cutoff_step=cutoff
        )
    return summary


def save_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def load_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["step"]


@hydra.main(version_base=None, config_path="../../conf", config_name="tsfm_moment")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    index = resolve_window_index(cfg)

    tags = [cfg.wandb.experiment, cfg.model, cfg.condition, cfg.experiment_kind]
    if cfg.experiment_kind == "controlled_scale":
        tags.append(f"assignment{cfg.scale_assignment}")
    wandb_run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=f"{cfg.model}-{cfg.condition}-{cfg.experiment_kind}"
        + (
            f"-{cfg.scale_assignment}"
            if cfg.experiment_kind == "controlled_scale"
            else ""
        ),
        group=f"{cfg.wandb.experiment}/{cfg.model}",
        job_type=cfg.condition,
        tags=tags,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    # Only meaningful when corpus.datasets is explicitly set: that's the one
    # case where we know exactly what was requested regardless of whether the
    # index came from a fresh build or a pre-built cache_path. When datasets
    # is null, "requested" nominally means "every corpus dataset", but a
    # loaded cache may have been built from a smaller --datasets restriction
    # (see scripts/build_gifteval_window_index.py) that this process has no
    # way to recover -- comparing against the full corpus discovery in that
    # case produces a bogus "N datasets contributed zero windows" warning for
    # datasets that were simply never part of this index's build, not
    # filtered out by window_length.
    zero_window_datasets = []
    if cfg.corpus.datasets is not None:
        zero_window_datasets = sorted(
            set(cfg.corpus.datasets) - set(index.table["dataset"].unique())
        )
    (out_dir / "window_index_meta.json").write_text(
        json.dumps(
            {
                "n_windows": len(index),
                "n_train": len(index.split("train")),
                "n_val": len(index.split("val")),
                "config": OmegaConf.to_container(cfg.window_index, resolve=True),
                "windows_per_dataset": index.table["dataset"].value_counts().to_dict(),
                "zero_window_datasets": zero_window_datasets,
            },
            indent=2,
        )
    )
    if zero_window_datasets:
        print(
            f"WARNING: {len(zero_window_datasets)} requested dataset(s) contributed zero "
            f"windows at context_length={cfg.window_index.context_length}+"
            f"prediction_length={cfg.window_index.prediction_length}: {zero_window_datasets}"
        )

    start = time.time()
    if cfg.model == "moment":
        summary = run_moment(cfg, index, wandb_run)
    elif cfg.model == "timesfm":
        summary = run_timesfm(cfg, index, wandb_run)
    else:
        raise ValueError(f"unknown model {cfg.model!r}")
    summary["wall_clock_seconds"] = time.time() - start

    print(json.dumps(summary, indent=2, default=str))
    wandb_run.finish()


if __name__ == "__main__":
    main()

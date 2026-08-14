"""Unified training entrypoint for the GiftEvalPretrain loss-space stages.

Runs exactly one (model, loss-space condition[, scale assignment]) combination
per process invocation and reports to Weights & Biases, matching this
project's existing convention (see src/loss_space/train.py) of one wandb run per setup,
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
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from src.configs import validate_config

from . import chronos2_adapter as ca
from . import gifteval_corpus as gc
from . import losses as L
from . import moirai2_adapter as m2
from . import moment_adapter as ma
from . import timesfm_model as tm
from . import window_index as wi
from .configs import TsfmConfig

TIMESFM_CONFIGS = {"17m": tm.CONFIG_17M, "70m": tm.CONFIG_70M}
# Fixed, not derived from cfg.seed -> the held-out eval sample (both the
# natural-mixture pooled draw and the per-dataset stratified draw) is
# identical across every checkpoint within a run and across every condition,
# matching src/loss_space/data.py's VAL_SEED convention ("the held-out validation set is
# identical across all runs"). Previously this was seeded by the current
# step, which resampled a different validation subset at every checkpoint --
# conflating real model progress with eval-sampling noise across the curve.
EVAL_SEED = 12345
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


def source_breakdown(
    per_example_loss: np.ndarray, rows, n_sources_min: int = 2, reducer: str = "mean"
) -> dict:
    """Gini / unweighted-mean at dataset, domain, and frequency granularity,
    per the plan's "Dispersion and equity metrics" section. Callers combine
    this with `pooled_global_error` computed separately -- from a natural-
    mixture sample, not this one, if `rows` came from
    `sample_stratified_eval_rows` (see that function's docstring for why)."""
    report = {}
    for level in ("dataset", "domain", "frequency"):
        group = (
            L.group_mean_by_source if reducer == "mean" else L.group_median_by_source
        )
        per_source = group(per_example_loss, rows[level].to_numpy())
        metrics = L.dispersion_metrics(per_source)
        if metrics["n_sources"] < n_sources_min:
            continue
        report[level] = {**metrics, "per_source_mean_error": per_source}
    return report


EVAL_CHUNK = 512  # windows per eval forward; bounds peak activation memory


def slice_batch(batch, start: int, stop: int):
    """Row-slices a MomentBatch/TimesFMBatch across its per-example fields,
    leaving scalar fields (e.g. batch_seed) untouched."""
    fields = {}
    for name, value in vars(batch).items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            fields[name] = value[start:stop]
        else:
            fields[name] = value
    return type(batch)(**fields)


def eval_scale_free(
    forward_fn,
    model,
    batch,
    condition: str,
    device: str,
    forward_kwargs: dict | None = None,
) -> tuple:
    """Per-example nMSE and MASE over `batch`, run in EVAL_CHUNK-sized forwards.

    The eval samples are eval_batches * batch_size windows, far too large to
    hold activations for in a single forward, so the batch is kept on CPU and
    moved chunk by chunk. Dropout is disabled for the duration: MOMENT trains
    with dropout 0.1, and sampling it at eval time adds noise to every
    reported metric.
    """
    forward_kwargs = {} if forward_kwargs is None else forward_kwargs
    was_training = model.training
    model.eval()
    n = len(batch.dataset)
    nmse_parts, mase_parts = [], []
    try:
        for start in range(0, n, EVAL_CHUNK):
            chunk = slice_batch(batch, start, min(start + EVAL_CHUNK, n)).to(device)
            with torch.no_grad():
                out = forward_fn(model, chunk, condition, **forward_kwargs)
            nmse_parts.append(out.normalized_mse.cpu().numpy())
            mase_parts.append(out.mase.cpu().numpy())
    finally:
        if was_training:
            model.train()
    return np.concatenate(nmse_parts), np.concatenate(mase_parts)


def log_dispersion(wandb_run, report: dict, step: int) -> None:
    flat = {
        "eval/pooled_global_error": report["pooled_global_error"],
        "eval/pooled_mase": report["pooled_mase"],
    }
    for level in ("dataset", "domain", "frequency"):
        if level not in report:
            continue
        flat[f"eval/{level}_gini"] = report[level]["gini"]
        flat[f"eval/{level}_unweighted_mean"] = report[level]["unweighted_mean"]
        flat[f"eval/{level}_n_sources"] = report[level]["n_sources"]
        for name, value in report[level]["per_source_mean_error"].items():
            flat[f"eval/{level}_error/{name}"] = value
        if level not in report["mase"]:
            continue
        flat[f"eval/mase_{level}_gini"] = report["mase"][level]["gini"]
        flat[f"eval/mase_{level}_unweighted_mean"] = report["mase"][level][
            "unweighted_mean"
        ]
    wandb_run.log(flat, step=step)


def sample_eval_rows(index: wi.WindowIndex, n: int, seed: int = EVAL_SEED):
    """A fixed (seed defaults to the module-level EVAL_SEED, not the current
    step) natural-mixture sample: dataset representation follows the val
    split's real, imbalanced proportions. Used for the pooled global metric,
    which is only informative relative to the unweighted per-source mean if
    it reflects the corpus's actual mixture rather than a rebalanced one --
    see the plan's "Dispersion and equity metrics" section. Not reliable for
    per-source breakdowns on its own: a dataset with few validation windows
    can easily draw zero in a given sample (see
    sample_stratified_eval_rows)."""
    val = index.split("val")
    if len(val) == 0:
        raise ValueError(
            "val split is empty; lower window_index.val_series_fraction is too small"
        )
    return val.sample(n=min(n, len(val)), random_state=seed)


def sample_stratified_eval_rows(
    index: wi.WindowIndex, n_per_dataset: int, seed: int = EVAL_SEED
) -> pd.DataFrame:
    """A fixed sample with up to `n_per_dataset` windows from EVERY dataset in
    the val split (fewer if a dataset's val pool is smaller), regardless of
    each dataset's natural size. Exists because the natural-mixture sample
    from `sample_eval_rows` can have a high probability of containing zero
    windows from a small dataset -- e.g. one real dataset in this corpus had
    only 7 total validation windows out of ~29,000, giving a ~61% chance of
    zero representation in a 2048-window natural sample (see
    notes/agentic_logs). Used only for the per-source Gini/unweighted-mean
    breakdown, never for the pooled global metric, since rebalancing there
    would make the pooled-vs-unweighted-mean comparison vacuous."""
    val = index.split("val")
    if len(val) == 0:
        raise ValueError(
            "val split is empty; lower window_index.val_series_fraction is too small"
        )
    parts = [
        group.sample(n=min(n_per_dataset, len(group)), random_state=seed)
        for _, group in val.groupby("dataset", sort=False)
    ]
    return pd.concat(parts)


def initialize_training_state(
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, dict, dict, dict, Path]:
    """Initialize a new run or continue one from a complete checkpoint."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.train.resume_from is None:
        return (
            0,
            windows_processed_counter(),
            {"steps_attempted": 0, "steps_skipped": 0},
            {"step": [], "pooled_mse": [], "reports": []},
            out_dir,
        )

    checkpoint_path = Path(cfg.train.resume_from)
    source_dir = checkpoint_path.parent
    if out_dir.resolve() == source_dir.resolve():
        raise ValueError("resume output_dir must differ from the source run directory")
    start_step = load_checkpoint(checkpoint_path, model, optimizer)
    if start_step >= cfg.train.steps:
        raise ValueError(
            f"checkpoint step {start_step} must be below train.steps {cfg.train.steps}"
        )

    history = json.loads((source_dir / "history.json").read_text())
    summary = json.loads((source_dir / "summary.json").read_text())
    if history["step"][-1] != start_step:
        raise ValueError("source history must end at the checkpoint step")
    optimization = {
        "steps_attempted": summary["optimization"]["steps_attempted"],
        "steps_skipped": summary["optimization"]["steps_skipped"],
    }
    if optimization["steps_attempted"] != start_step:
        raise ValueError("source optimization count must equal the checkpoint step")
    return (
        start_step,
        summary["windows_processed"],
        optimization,
        history,
        out_dir,
    )


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
        cfg.train.steps
        if cfg.train.schedule_steps is None
        else cfg.train.schedule_steps,
        cfg.train.batch_size,
        cfg.train.schedule_seed,
    )
    train_table = index.split("train").reset_index(drop=True)
    scale_assignment = (
        cfg.scale_assignment if cfg.experiment_kind == "controlled_scale" else None
    )

    # Fixed once for the whole run (see EVAL_SEED) so every checkpoint's
    # metrics are comparable to every other checkpoint's, and so paired
    # conditions evaluate on identical eval windows too.
    pooled_eval_rows = sample_eval_rows(
        index, cfg.train.eval_batches * cfg.train.batch_size
    )
    # No scale_assignment: the controlled scale b is a TRAINING-time
    # intervention, so every arm is evaluated on identical natural-scale
    # windows. Kept on CPU and moved chunk by chunk (see eval_scale_free).
    pooled_eval_batch = ma.make_batch(index, pooled_eval_rows, cache)
    strat_eval_rows = sample_stratified_eval_rows(
        index, cfg.train.eval_windows_per_dataset
    )
    strat_eval_batch = ma.make_batch(index, strat_eval_rows, cache)

    start_step, counter, optimization, history, out_dir = initialize_training_state(
        cfg, model, optimizer
    )

    for step in range(start_step + 1, cfg.train.steps + 1):
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
        optimization["steps_attempted"] += 1
        optimization["steps_skipped"] += int(metrics["step_skipped"])
        update_windows_processed(counter, rows)

        wandb_run.log(
            {
                "train/loss": metrics["loss"],
                "train/grad_norm_before_clip": metrics["grad_norm_before_clip"],
                "train/grad_norm_after_clip": metrics["grad_norm_after_clip"],
                "train/clipped": float(metrics["clipped"]),
                "train/step_skipped": float(metrics["step_skipped"]),
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
            # Evaluation is always scale-free (nMSE primary, MASE secondary)
            # regardless of which space `condition` optimizes in, so both arms
            # are scored on identical definitions -- see results.tex, which
            # reports nMSE throughout.
            pooled_nmse, pooled_mase = eval_scale_free(
                ma.forward, model, pooled_eval_batch, cfg.condition, cfg.device
            )
            strat_error, strat_mase = eval_scale_free(
                ma.forward, model, strat_eval_batch, cfg.condition, cfg.device
            )
            report = {
                "pooled_global_error": L.pooled_mean(pooled_nmse),
                "pooled_mase": L.pooled_mean(pooled_mase),
                **source_breakdown(strat_error, strat_eval_rows),
                "mase": source_breakdown(strat_mase, strat_eval_rows),
            }
            log_dispersion(wandb_run, report, step)
            history["step"].append(step)
            history["pooled_mse"].append(report["pooled_global_error"])
            history["reports"].append(report)
            (out_dir / "history.json").write_text(
                json.dumps(history, indent=2, default=str)
            )
            for level in ("dataset", "domain", "frequency"):
                wandb_run.log(
                    {f"exposure/{level}_windows_processed_n": len(counter[level])},
                    step=step,
                )

        if step % cfg.train.checkpoint_every == 0 or step == cfg.train.steps:
            save_checkpoint(
                out_dir / f"checkpoint_step{step}.pt", model, optimizer, step
            )

    summary = finalize_summary(history, counter, cfg, optimization)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
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
        cfg.train.steps
        if cfg.train.schedule_steps is None
        else cfg.train.schedule_steps,
        cfg.train.batch_size,
        cfg.train.schedule_seed,
    )
    train_table = index.split("train").reset_index(drop=True)
    scale_assignment = (
        cfg.scale_assignment if cfg.experiment_kind == "controlled_scale" else None
    )

    # Fixed once for the whole run (see EVAL_SEED) so every checkpoint's
    # metrics are comparable to every other checkpoint's, and so paired
    # conditions evaluate on identical eval windows too.
    pooled_eval_rows = sample_eval_rows(
        index, cfg.train.eval_batches * cfg.train.batch_size
    )
    # Natural scale only: b is a training-time intervention (see run_moment).
    pooled_eval_batch = tm.make_batch(
        index, pooled_eval_rows, cache, model_config.horizon_len
    )
    strat_eval_rows = sample_stratified_eval_rows(
        index, cfg.train.eval_windows_per_dataset
    )
    strat_eval_batch = tm.make_batch(
        index, strat_eval_rows, cache, model_config.horizon_len
    )

    start_step, counter, optimization, history, out_dir = initialize_training_state(
        cfg, model, optimizer
    )

    for step in range(start_step + 1, cfg.train.steps + 1):
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
            model,
            batch,
            cfg.condition,
            cfg.timesfm.normalization_mode,
            cfg.timesfm.objective,
            optimizer,
            cfg.timesfm.grad_clip_norm,
        )
        optimization["steps_attempted"] += 1
        optimization["steps_skipped"] += int(metrics["step_skipped"])
        update_windows_processed(counter, rows)

        wandb_run.log(
            {
                "train/loss": metrics["loss"],
                "train/mse": float(np.mean(metrics["mse_per_example"])),
                "train/pinball": float(np.mean(metrics["pinball_per_example"])),
                "train/mse_to_pinball_loss_ratio": metrics["mse_to_pinball_loss_ratio"],
                "train/output_head_grad_norm": metrics["output_head_grad_norm"],
                "train/total_grad_norm_before_clip": metrics[
                    "total_grad_norm_before_clip"
                ],
                "train/total_grad_norm_after_clip": metrics[
                    "total_grad_norm_after_clip"
                ],
                "train/clipped": float(metrics["clipped"]),
                "train/step_skipped": float(metrics["step_skipped"]),
                "train/degenerate_frac": metrics["degenerate_frac"],
            },
            step=step,
        )

        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            # Scale-free evaluation regardless of `condition`; see run_moment.
            pooled_nmse, pooled_mase = eval_scale_free(
                tm.forward,
                model,
                pooled_eval_batch,
                cfg.condition,
                cfg.device,
                {"normalization_mode": cfg.timesfm.normalization_mode},
            )
            strat_error, strat_mase = eval_scale_free(
                tm.forward,
                model,
                strat_eval_batch,
                cfg.condition,
                cfg.device,
                {"normalization_mode": cfg.timesfm.normalization_mode},
            )
            report = {
                "pooled_global_error": L.pooled_mean(pooled_nmse),
                "pooled_mase": L.pooled_mean(pooled_mase),
                **source_breakdown(strat_error, strat_eval_rows),
                "mase": source_breakdown(strat_mase, strat_eval_rows),
            }
            log_dispersion(wandb_run, report, step)
            history["step"].append(step)
            history["pooled_mse"].append(report["pooled_global_error"])
            history["reports"].append(report)
            (out_dir / "history.json").write_text(
                json.dumps(history, indent=2, default=str)
            )
            for level in ("dataset", "domain", "frequency"):
                wandb_run.log(
                    {f"exposure/{level}_windows_processed_n": len(counter[level])},
                    step=step,
                )

        if step % cfg.train.checkpoint_every == 0 or step == cfg.train.steps:
            save_checkpoint(
                out_dir / f"checkpoint_step{step}.pt", model, optimizer, step
            )

    summary = finalize_summary(history, counter, cfg, optimization)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_chronos2(cfg: DictConfig, index: wi.WindowIndex, wandb_run) -> dict:
    if cfg.condition not in ca.CONDITIONS:
        raise ValueError(f"condition must be one of {ca.CONDITIONS}")
    model_config = ca.Chronos2Config(
        **OmegaConf.to_container(cfg.chronos2, resolve=True)
    )
    if model_config.context_length != cfg.window_index.context_length:
        raise ValueError(
            "chronos2.context_length must equal window_index.context_length"
        )
    if model_config.prediction_length != cfg.window_index.prediction_length:
        raise ValueError(
            "chronos2.prediction_length must equal window_index.prediction_length"
        )

    dataset_weights = resolve_dataset_weights(cfg, index)
    model = ca.build_chronos2_model(model_config, seed=cfg.seed).to(cfg.device)
    optimizer = OPTIMIZERS[cfg.train.optimizer](model.parameters(), lr=cfg.train.lr)
    cache = wi.SeriesCache(index.corpus_root)
    schedule = wi.build_batch_schedule(
        index,
        "train",
        dataset_weights,
        cfg.train.steps
        if cfg.train.schedule_steps is None
        else cfg.train.schedule_steps,
        cfg.train.batch_size,
        cfg.train.schedule_seed,
    )
    train_table = index.split("train").reset_index(drop=True)
    scale_assignment = (
        cfg.scale_assignment if cfg.experiment_kind == "controlled_scale" else None
    )

    pooled_eval_rows = sample_eval_rows(
        index, cfg.train.eval_batches * cfg.train.batch_size
    )
    pooled_eval_batch = ca.make_batch(index, pooled_eval_rows, cache)
    strat_eval_rows = sample_stratified_eval_rows(
        index, cfg.train.eval_windows_per_dataset
    )
    strat_eval_batch = ca.make_batch(index, strat_eval_rows, cache)

    start_step, counter, optimization, history, out_dir = initialize_training_state(
        cfg, model, optimizer
    )

    for step in range(start_step + 1, cfg.train.steps + 1):
        rows = train_table.iloc[schedule[step - 1]]
        batch = ca.make_batch(
            index,
            rows,
            cache,
            scale_assignment=scale_assignment,
            b_low=cfg.scale_b_low,
            b_high=cfg.scale_b_high,
        ).to(cfg.device)
        metrics = ca.training_step_metrics(
            model,
            batch,
            cfg.condition,
            optimizer,
            model_config.grad_clip_norm,
        )
        optimization["steps_attempted"] += 1
        optimization["steps_skipped"] += int(metrics["step_skipped"])
        update_windows_processed(counter, rows)

        wandb_run.log(
            {
                "train/loss": metrics["loss"],
                "train/output_head_grad_norm": metrics["output_head_grad_norm"],
                "train/total_grad_norm_before_clip": metrics[
                    "total_grad_norm_before_clip"
                ],
                "train/total_grad_norm_after_clip": metrics[
                    "total_grad_norm_after_clip"
                ],
                "train/clipped": float(metrics["clipped"]),
                "train/step_skipped": float(metrics["step_skipped"]),
                "train/degenerate_frac": metrics["degenerate_frac"],
            },
            step=step,
        )

        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            pooled_nmse, pooled_mase = eval_scale_free(
                ca.forward,
                model,
                pooled_eval_batch,
                cfg.condition,
                cfg.device,
            )
            strat_error, strat_mase = eval_scale_free(
                ca.forward,
                model,
                strat_eval_batch,
                cfg.condition,
                cfg.device,
            )
            report = {
                "pooled_global_error": L.pooled_mean(pooled_nmse),
                "pooled_mase": L.pooled_mean(pooled_mase),
                **source_breakdown(strat_error, strat_eval_rows),
                "mase": source_breakdown(strat_mase, strat_eval_rows),
            }
            log_dispersion(wandb_run, report, step)
            history["step"].append(step)
            history["pooled_mse"].append(report["pooled_global_error"])
            history["reports"].append(report)
            (out_dir / "history.json").write_text(
                json.dumps(history, indent=2, default=str)
            )
            for level in ("dataset", "domain", "frequency"):
                wandb_run.log(
                    {f"exposure/{level}_windows_processed_n": len(counter[level])},
                    step=step,
                )

        if step % cfg.train.checkpoint_every == 0 or step == cfg.train.steps:
            save_checkpoint(
                out_dir / f"checkpoint_step{step}.pt", model, optimizer, step
            )

    summary = finalize_summary(history, counter, cfg, optimization)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_moirai2(cfg: DictConfig, index: wi.WindowIndex, wandb_run) -> dict:
    if cfg.condition not in m2.CONDITIONS:
        raise ValueError(f"condition must be one of {m2.CONDITIONS}")
    model_config = m2.Moirai2Config(**OmegaConf.to_container(cfg.moirai2, resolve=True))
    if model_config.context_length != cfg.window_index.context_length:
        raise ValueError(
            "moirai2.context_length must equal window_index.context_length"
        )
    if model_config.predict_horizon > cfg.window_index.prediction_length:
        raise ValueError(
            "moirai2.predict_horizon must not exceed window_index.prediction_length"
        )

    dataset_weights = resolve_dataset_weights(cfg, index)
    model = m2.build_moirai2_model(model_config, seed=cfg.seed).to(cfg.device)
    optimizer = OPTIMIZERS[cfg.train.optimizer](model.parameters(), lr=cfg.train.lr)
    cache = wi.SeriesCache(index.corpus_root)
    schedule = wi.build_batch_schedule(
        index,
        "train",
        dataset_weights,
        cfg.train.steps
        if cfg.train.schedule_steps is None
        else cfg.train.schedule_steps,
        cfg.train.batch_size,
        cfg.train.schedule_seed,
    )
    train_table = index.split("train").reset_index(drop=True)
    scale_assignment = (
        cfg.scale_assignment if cfg.experiment_kind == "controlled_scale" else None
    )

    pooled_eval_rows = sample_eval_rows(
        index, cfg.train.eval_batches * cfg.train.batch_size
    )
    pooled_eval_batch = m2.make_batch(index, pooled_eval_rows, cache, model_config)
    strat_eval_rows = sample_stratified_eval_rows(
        index, cfg.train.eval_windows_per_dataset
    )
    strat_eval_batch = m2.make_batch(index, strat_eval_rows, cache, model_config)

    start_step, counter, optimization, history, out_dir = initialize_training_state(
        cfg, model, optimizer
    )

    for step in range(start_step + 1, cfg.train.steps + 1):
        rows = train_table.iloc[schedule[step - 1]]
        batch = m2.make_batch(
            index,
            rows,
            cache,
            model_config,
            scale_assignment=scale_assignment,
            b_low=cfg.scale_b_low,
            b_high=cfg.scale_b_high,
        ).to(cfg.device)
        metrics = m2.training_step_metrics(
            model,
            batch,
            cfg.condition,
            model_config,
            optimizer,
            model_config.grad_clip_norm,
        )
        optimization["steps_attempted"] += 1
        optimization["steps_skipped"] += int(metrics["step_skipped"])
        update_windows_processed(counter, rows)

        wandb_run.log(
            {
                "train/loss": metrics["loss"],
                "train/output_head_grad_norm": metrics["output_head_grad_norm"],
                "train/total_grad_norm_before_clip": metrics[
                    "total_grad_norm_before_clip"
                ],
                "train/total_grad_norm_after_clip": metrics[
                    "total_grad_norm_after_clip"
                ],
                "train/clipped": float(metrics["clipped"]),
                "train/step_skipped": float(metrics["step_skipped"]),
                "train/degenerate_frac": metrics["degenerate_frac"],
            },
            step=step,
        )

        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            pooled_nmse, pooled_mase = eval_scale_free(
                m2.forward,
                model,
                pooled_eval_batch,
                cfg.condition,
                cfg.device,
                {"config": model_config},
            )
            strat_error, strat_mase = eval_scale_free(
                m2.forward,
                model,
                strat_eval_batch,
                cfg.condition,
                cfg.device,
                {"config": model_config},
            )
            report = {
                "pooled_global_error": L.pooled_mean(pooled_nmse),
                "pooled_mase": L.pooled_mean(pooled_mase),
                **source_breakdown(strat_error, strat_eval_rows),
                "mase": source_breakdown(strat_mase, strat_eval_rows),
            }
            log_dispersion(wandb_run, report, step)
            history["step"].append(step)
            history["pooled_mse"].append(report["pooled_global_error"])
            history["reports"].append(report)
            (out_dir / "history.json").write_text(
                json.dumps(history, indent=2, default=str)
            )
            for level in ("dataset", "domain", "frequency"):
                wandb_run.log(
                    {f"exposure/{level}_windows_processed_n": len(counter[level])},
                    step=step,
                )

        if step % cfg.train.checkpoint_every == 0 or step == cfg.train.steps:
            save_checkpoint(
                out_dir / f"checkpoint_step{step}.pt", model, optimizer, step
            )

    summary = finalize_summary(history, counter, cfg, optimization)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def finalize_summary(
    history: dict, counter: dict, cfg: DictConfig, optimization: dict
) -> dict:
    steps = np.array(history["step"])
    mse = np.array(history["pooled_mse"])
    summary = {
        "windows_processed": counter,
        "optimization": {
            **optimization,
            "updates_applied": optimization["steps_attempted"]
            - optimization["steps_skipped"],
        },
        "final_pooled_mse": float(mse[-1]) if mse.size else None,
        "moment_revision": ma.MOMENT_REVISION if cfg.model == "moment" else None,
        "timesfm_revision": tm.TIMESFM_REVISION if cfg.model == "timesfm" else None,
        "chronos2_revision": (
            ca.CHRONOS2_REVISION if cfg.model == "chronos2" else None
        ),
        "moirai2_revision": (m2.MOIRAI2_REVISION if cfg.model == "moirai2" else None),
    }
    early = steps <= L.TROUGH_STEP_CUTOFF
    if early.sum() >= 2:
        summary["log_mse_auc_through_2000"] = L.log_mse_auc(steps, mse)
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
    validate_config(cfg, TsfmConfig)
    if cfg.train.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
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
    # (see src/tsfm_pretraining/scripts/build_gifteval_window_index.py)
    # that this process has no way to recover -- comparing against the full corpus discovery in that
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
                "dataset_scale_groups": index.dataset_scale_groups(),
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
    elif cfg.model == "chronos2":
        summary = run_chronos2(cfg, index, wandb_run)
    elif cfg.model == "moirai2":
        summary = run_moirai2(cfg, index, wandb_run)
    else:
        raise ValueError(f"unknown model {cfg.model!r}")
    summary["wall_clock_seconds"] = time.time() - start

    print(json.dumps(summary, indent=2, default=str))
    wandb_run.finish()


if __name__ == "__main__":
    main()

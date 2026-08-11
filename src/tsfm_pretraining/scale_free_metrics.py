"""Recompute scale-free eval metrics (nMSE, MASE) from saved checkpoints.

The original GiftEvalPretrain runs scored each variant on `per_example_loss_masked`
(MOMENT) / `mse_per_example` (TimesFM), i.e. the loss in whichever space that
variant trained in. That makes the normalized and original variants incomparable to
each other, and makes the reported per-dataset Gini a function of each
dataset's physical units rather than of model quality. This module rereads the
saved checkpoints and rescores every run on identical, scale-free definitions,
matching the nMSE convention used throughout overleaf/sections/results.tex.

Every run is evaluated on the natural (unscaled) held-out windows regardless of
the scale assignment it trained under, so all variants are scored on exactly the
same data. The controlled scale b is a training-time intervention only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.tsfm_pretraining import losses as L
from src.tsfm_pretraining import moment_adapter as ma
from src.tsfm_pretraining import timesfm_model as tm
from src.tsfm_pretraining import train as T
from src.tsfm_pretraining import window_index as wi


def build_run_context(
    cfg, index, cache, device: str, chunk_size: int, pooled_windows: int
) -> dict:
    """Everything that is fixed across a run's checkpoints: the model skeleton
    and the two eval batches, built once. The eval rows use the same fixed
    EVAL_SEED the runs trained against, and no scale assignment is applied, so
    every variant is scored on identical natural-scale windows."""
    strat_rows = T.sample_stratified_eval_rows(
        index, cfg.train.eval_windows_per_dataset
    )
    pooled_rows = T.sample_eval_rows(
        index, min(pooled_windows, cfg.train.eval_batches * cfg.train.batch_size)
    )

    if cfg.model == "moment":
        model_cfg = ma.MomentConfig(**OmegaConf.to_container(cfg.moment, resolve=True))
        model = ma.build_moment_model(model_cfg, seed=cfg.seed).to(device)
        make_args, make, fwd = (), ma.make_batch, ma.forward
    else:
        model_cfg = T.TIMESFM_CONFIGS[cfg.timesfm.config_size]
        model = tm.build_timesfm_model(model_cfg, seed=cfg.seed).to(device)
        make_args, make, fwd = (model_cfg.horizon_len,), tm.make_batch, tm.forward

    # Kept on CPU and moved chunk-by-chunk: the pooled sample is
    # eval_batches * batch_size windows, too large to hold activations for in
    # a single forward. Building them reads every window's series off disk and
    # dominates a run's wall clock, so announce it rather than sitting silent.
    print(
        f"building eval batches: {len(strat_rows)} stratified + "
        f"{len(pooled_rows)} pooled windows",
        flush=True,
    )
    return {
        "model": model,
        "forward": fwd,
        "condition": cfg.condition,
        "chunk_size": chunk_size,
        "strat_rows": strat_rows,
        "strat_batch": make(index, strat_rows, cache, *make_args),
        "pooled_batch": make(index, pooled_rows, cache, *make_args),
    }


def slice_batch(batch, start: int, stop: int):
    """Row-slices a MomentBatch/TimesFMBatch across all of its per-example
    fields, leaving scalar fields (e.g. batch_seed) untouched."""
    fields = {}
    for name, value in vars(batch).items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            fields[name] = value[start:stop]
        else:
            fields[name] = value
    return type(batch)(**fields)


def forward_chunked(ctx: dict, batch, device: str) -> tuple:
    """Runs the adapter forward in chunks and concatenates per-example nMSE and
    MASE, so peak activation memory is bounded by chunk_size rather than by the
    full eval sample.

    MOMENT's reconstruction mask is drawn from `batch_seed` at the batch's own
    shape, so chunking yields a different mask than one full-batch forward
    would. That is deterministic in (batch_seed, chunk_size) and chunk
    boundaries are identical across every run and checkpoint here (the eval
    rows come from the same fixed EVAL_SEED), so all variants remain directly
    comparable to each other -- they are simply not bitwise comparable to the
    original training-time eval numbers, which were on the wrong metric anyway.
    """
    n = len(batch.dataset)
    nmse_parts, mase_parts = [], []
    for start in range(0, n, ctx["chunk_size"]):
        stop = min(start + ctx["chunk_size"], n)
        chunk = slice_batch(batch, start, stop).to(device)
        with torch.no_grad():
            out = ctx["forward"](ctx["model"], chunk, ctx["condition"])
        nmse_parts.append(out.normalized_mse.cpu().numpy())
        mase_parts.append(out.mase.cpu().numpy())
    return np.concatenate(nmse_parts), np.concatenate(mase_parts)


def evaluate_checkpoint(ctx: dict, ckpt_path: Path, device: str) -> dict:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ctx["model"].load_state_dict(state["model"])
    ctx["model"].eval()

    strat_nmse, strat_mase = forward_chunked(ctx, ctx["strat_batch"], device)
    pooled_nmse, pooled_mase = forward_chunked(ctx, ctx["pooled_batch"], device)
    return {
        "step": int(state["step"]),
        "strat_nmse": strat_nmse,
        "strat_mase": strat_mase,
        "pooled_nmse": L.pooled_mean(pooled_nmse),
        "pooled_mase": L.pooled_mean(pooled_mase),
    }


def recompute_metrics(
    run_entries: list[str],
    output_dir: Path,
    device: str,
    chunk_size: int,
    pooled_windows: int,
) -> None:
    """Loads every `label=path` entry in run_entries, rescores each run's
    checkpoints on scale-free nMSE/MASE, and writes scale_free_metrics.csv and
    final_per_dataset.json to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_out = []
    per_dataset_out = {}
    # Every run draws its eval rows from the same window index with the same
    # fixed EVAL_SEED and applies no scale, so the batches depend only on the
    # model family's window/horizon geometry -- build them once per family
    # rather than re-reading every window's series off disk for all 12 runs.
    contexts: dict[str, dict] = {}
    for entry in run_entries:
        label, path = entry.split("=", 1)
        run_dir = Path(path)
        cfg = OmegaConf.load(run_dir / "resolved_config.yaml")

        checkpoints = sorted(
            run_dir.glob("checkpoint_step*.pt"),
            key=lambda p: int(p.stem.removeprefix("checkpoint_step")),
        )
        if not checkpoints:
            raise FileNotFoundError(f"no checkpoints in {run_dir}")

        key = str(cfg.model)
        if key not in contexts:
            index = T.resolve_window_index(cfg)
            cache = wi.SeriesCache(index.corpus_root)
            contexts[key] = build_run_context(
                cfg, index, cache, device, chunk_size, pooled_windows
            )
        ctx = contexts[key]
        # The loss space only selects which term the objective used; the
        # evaluation metrics are computed unconditionally, so reusing one
        # model skeleton across a family's conditions is safe.
        ctx["condition"] = cfg.condition
        final_reports = None
        for ckpt in checkpoints:
            ev = evaluate_checkpoint(ctx, ckpt, device)
            nmse_report = T.source_breakdown(ev["strat_nmse"], ctx["strat_rows"])
            mase_report = T.source_breakdown(ev["strat_mase"], ctx["strat_rows"])
            # Per-window nMSE is heavy-tailed on sparse real series (one m5
            # window reaches ~1e7 because its context std is far below a rare
            # spike), so a median-reduced variant is carried alongside.
            nmse_med_report = T.source_breakdown(
                ev["strat_nmse"], ctx["strat_rows"], reducer="median"
            )
            final_reports = (nmse_report, mase_report, nmse_med_report)
            rows_out.append(
                {
                    "label": label,
                    "step": ev["step"],
                    "pooled_nmse": ev["pooled_nmse"],
                    "pooled_mase": ev["pooled_mase"],
                    "nmse_dataset_gini": nmse_report["dataset"]["gini"],
                    "nmse_dataset_mean": nmse_report["dataset"]["unweighted_mean"],
                    "mase_dataset_gini": mase_report["dataset"]["gini"],
                    "mase_dataset_mean": mase_report["dataset"]["unweighted_mean"],
                    "mase_n_sources": mase_report["dataset"]["n_sources"],
                    "nmse_median_dataset_gini": nmse_med_report["dataset"]["gini"],
                    "nmse_median_dataset_mean": nmse_med_report["dataset"][
                        "unweighted_mean"
                    ],
                    "nmse_domain_gini": nmse_report["domain"]["gini"],
                    "mase_domain_gini": mase_report["domain"]["gini"],
                }
            )
            print(f"{label} step={ev['step']} done", flush=True)

        per_dataset_out[label] = {
            "nmse": final_reports[0]["dataset"]["per_source_mean_error"],
            "mase": final_reports[1]["dataset"]["per_source_mean_error"],
            "nmse_median": final_reports[2]["dataset"]["per_source_mean_error"],
        }

    table = pd.DataFrame(rows_out)
    table.to_csv(output_dir / "scale_free_metrics.csv", index=False)
    (output_dir / "final_per_dataset.json").write_text(
        json.dumps(per_dataset_out, indent=2)
    )
    final = table.sort_values("step").groupby("label").tail(1)
    print(final.to_string(index=False))

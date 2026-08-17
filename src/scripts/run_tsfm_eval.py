"""Scores a trained checkpoint against the six held-out evaluation suites.

Writes one row per (suite, mode, subset) plus a pooled row per (suite, mode)
to `output_dir`. The three modes are never pooled together: they measure
different things over different subsets of each suite, so mixing them would
produce a number that means nothing.

Usage:
  uv run python -m src.scripts.run_tsfm_eval \
      checkpoint=/path/to/checkpoint_step100000.pt model=timesfm
"""

import collections
import json
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.eval import predict, score, suites
from src.eval.protocol import build_forecaster
from src.metrics import accuracy


def score_native(forecaster, series, batch_size: int) -> list[dict]:
    """Each subset's own published protocol, the headline numbers."""
    rows = []
    for subset, items in _by_subset(series).items():
        forecasts = predict.run(forecaster, items, batch_size=batch_size)
        metrics = score.score(forecasts, items)
        rows.append(
            {
                "subset": subset,
                "freq": _freq(items),
                "n_series": len(items),
                **_pool(metrics),
            }
        )
    return rows


def score_fixed(forecaster, series, batch_size: int) -> list[dict]:
    """The training window shape, as a cross-check on the native path.

    Reports `n_dropped` next to every row. On M1, M3, and Tourism that is
    every series and the mode is empty, which is the finding rather than a
    failure.
    """
    rows = []
    for subset, items in _by_subset(series).items():
        kept, dropped = predict.as_fixed_windows(
            items, forecaster.context_length, forecaster.horizon
        )
        if not kept:
            rows.append(
                {
                    "subset": subset,
                    "freq": _freq(items),
                    "n_series": 0,
                    "n_dropped": dropped,
                }
            )
            continue
        forecasts = predict.run(forecaster, kept, batch_size=batch_size)
        metrics = score.score(forecasts, kept)
        rows.append(
            {
                "subset": subset,
                "freq": _freq(items),
                "n_series": len(kept),
                "n_dropped": dropped,
                **_pool(metrics),
            }
        )
    return rows


def score_rolling(forecaster, series, cfg) -> list[dict]:
    """Excess Volatility and sFPC over overlapping re-forecasts.

    Capped per subset because every series is forecast `n_windows` times, and
    subsets too short for the window stack are reported as skipped rather
    than silently omitted.
    """
    rows = []
    horizon = min(int(cfg.rolling.horizon), forecaster.horizon)
    needed = horizon + (int(cfg.rolling.n_windows) - 1) * int(cfg.rolling.stride) + 1
    for subset, items in _by_subset(series).items():
        usable = [
            item for item in items if len(item.history) + len(item.actual) >= needed
        ]
        if not usable:
            rows.append(
                {
                    "subset": subset,
                    "freq": _freq(items),
                    "n_series": 0,
                    "skipped": "too short",
                }
            )
            continue
        sample = usable[: int(cfg.rolling.max_series_per_subset)]
        rolling = predict.run_rolling(
            forecaster,
            sample,
            horizon=horizon,
            stride=int(cfg.rolling.stride),
            n_windows=int(cfg.rolling.n_windows),
            batch_size=int(cfg.batch_size),
        )
        rows.append(
            {
                "subset": subset,
                "freq": _freq(items),
                "n_series": len(sample),
                "n_too_short": len(items) - len(usable),
                **score.score_stability(rolling),
            }
        )
    return rows


def _by_subset(series):
    grouped = collections.defaultdict(list)
    for item in series:
        grouped[item.subset].append(item)
    return dict(sorted(grouped.items()))


def _freq(items) -> str:
    """The frequency label a subset is reported under.

    Subsets are frequency-homogeneous in every suite (GIFT-Eval keys its
    configs as `dataset/frequency`, and the Monash and M4 suites split by
    frequency outright), so a subset carrying more than one is a loader bug
    rather than something to average over. M3 Other declares no frequency.
    """
    labels = sorted({item.freq for item in items if item.freq is not None})
    if not labels:
        return "none"
    if len(labels) > 1:
        raise ValueError(f"subset {items[0].subset!r} mixes frequencies {labels}")
    return labels[0]


def _pool(metrics):
    return accuracy.pool(metrics)


def adopt_run_config(cfg: DictConfig) -> None:
    """Replaces the model skeleton with the one the checkpoint was trained on.

    A checkpoint only loads into the architecture that produced it, so the
    skeleton in `conf/eval.yaml` is a default rather than a source of truth.
    Taking `model` and its block from the run's own `resolved_config.yaml`
    removes the chance of scoring a checkpoint under a skeleton that merely
    happens to load.
    """
    resolved = Path(cfg.run_dir) / "resolved_config.yaml"
    if not resolved.is_file():
        raise FileNotFoundError(f"no resolved_config.yaml in {cfg.run_dir}")
    run_cfg = OmegaConf.load(resolved)
    cfg.model = run_cfg.model
    cfg[cfg.model] = run_cfg[cfg.model]
    cfg.condition = run_cfg.condition
    cfg.seed = run_cfg.seed
    print(f"model skeleton adopted from {resolved}", flush=True)


@hydra.main(version_base=None, config_path="../../conf", config_name="eval")
def main(cfg: DictConfig) -> None:
    if cfg.run_dir is not None:
        adopt_run_config(cfg)
    checkpoint = Path(cfg.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    forecaster = build_forecaster(cfg, checkpoint, cfg.device)
    print(
        f"{cfg.model}: context {forecaster.context_length}, native horizon "
        f"{forecaster.horizon}, {len(forecaster.quantiles)} quantiles",
        flush=True,
    )

    roots = OmegaConf.to_container(cfg.roots, resolve=True)
    records = []
    for suite in cfg.suites:
        series = suites.load_suite(suite, roots)
        print(f"{suite}: {len(series)} series", flush=True)
        for mode in cfg.modes:
            if mode == "native":
                rows = score_native(forecaster, series, int(cfg.batch_size))
            elif mode == "fixed":
                rows = score_fixed(forecaster, series, int(cfg.batch_size))
            elif mode == "rolling":
                rows = score_rolling(forecaster, series, cfg)
            else:
                raise ValueError(f"unknown mode {mode!r}")
            for row in rows:
                records.append({"suite": suite, "mode": mode, **row})
            print(f"  {mode}: {len(rows)} subsets scored", flush=True)

    table = pd.DataFrame(records)
    # Identity travels in the rows so the collector can concatenate the runs
    # without parsing directory names back into condition and seed.
    table.insert(0, "seed", cfg.seed)
    table.insert(0, "condition", cfg.condition)
    table.insert(0, "model", cfg.model)
    table.to_csv(out_dir / "eval_by_subset.csv", index=False)

    # Suite-level rows weight subsets by series count, and the per-subset CSV
    # sits alongside because the suites are unbalanced enough that a pooled
    # mean can be close to a report on a single large config.
    summary = {}
    for (suite, mode), group in table.groupby(["suite", "mode"]):
        counts = group.get("n_series", pd.Series(dtype=float)).fillna(0)
        total = float(counts.sum())
        entry = {"n_series": total, "n_subsets": len(group)}
        for column in group.columns:
            if column in ("suite", "mode", "subset", "skipped") or total == 0:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            usable = values.notna() & (counts > 0)
            if usable.any():
                entry[column] = float(
                    (values[usable] * counts[usable]).sum() / counts[usable].sum()
                )
        summary[f"{suite}/{mode}"] = entry
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nwrote {out_dir / 'eval_by_subset.csv'}")
    print(f"wrote {out_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()

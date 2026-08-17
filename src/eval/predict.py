"""Runs a `Forecaster` over a loaded eval suite.

Holds no model-specific code. Its job is turning ragged `EvalSeries` into
the fixed-width left-padded batches every adapter expects, and asserting the
count survives, which is the guard that keeps the suite definitions honest.
"""

from dataclasses import dataclass

import numpy as np

from src.eval.protocol import Forecaster
from src.eval.suites import EvalSeries


@dataclass
class Forecasts:
    """Aligned arrays over a suite, in the order the series were given.

    `quantiles` are the levels of `values`' last axis. `history` and
    `history_mask` are the left-padded context actually shown to the model,
    returned so the scorer derives MASE and nMSE denominators from exactly
    that rather than re-deriving them from the raw series.
    """

    values: np.ndarray  # [N, H, Q]
    actual: np.ndarray  # [N, H]
    actual_mask: np.ndarray  # [N, H]
    history: np.ndarray  # [N, L]
    history_mask: np.ndarray  # [N, L]
    periods: np.ndarray  # [N]
    quantiles: list[float]
    subsets: list[str]
    item_ids: list[str]


def build_context(
    series: list[EvalSeries], context_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Left-pads or truncates each history to `context_length`.

    Series shorter than the context are padded on the left and masked, never
    dropped. That is the whole point of the native mode: 100% of M1, M3, and
    Tourism and 95% of M4 are shorter than a 512 context, so a length filter
    here would silently empty three suites (see
    notes/agentic_logs/2026-08-16-eval-harness.md).
    """
    context = np.zeros((len(series), context_length), dtype=np.float64)
    valid = np.zeros((len(series), context_length), dtype=np.float64)
    for i, item in enumerate(series):
        history = item.history[-context_length:]
        observed = np.isfinite(history)
        context[i, context_length - len(history) :] = np.nan_to_num(history, nan=0.0)
        valid[i, context_length - len(history) :] = observed
    return context, valid


def as_fixed_windows(
    series: list[EvalSeries], context_length: int, horizon: int
) -> tuple[list[EvalSeries], int]:
    """Re-cuts each series to a full context plus horizon from its tail.

    This is the one place in the harness that drops series, and it does so
    deliberately: the `fixed` mode exists to reproduce the training-time
    window shape as a cross-check, so a series shorter than
    context + horizon has no such window. The count dropped is returned and
    must be reported next to the numbers, since on M1, M3, and Tourism it is
    every series and the mode yields nothing at all.
    """
    needed = context_length + horizon
    kept = []
    for item in series:
        full = np.concatenate([item.history, item.actual])
        if len(full) < needed:
            continue
        kept.append(
            EvalSeries(
                suite=item.suite,
                subset=item.subset,
                item_id=item.item_id,
                history=full[-needed:-horizon],
                actual=full[-horizon:],
                period=item.period,
                freq=item.freq,
            )
        )
    return kept, len(series) - len(kept)


@dataclass
class RollingForecasts:
    """Overlapping rolling forecasts shaped for the stability metrics.

    The window axis is ordered **oldest-first**, window t created before
    t+1, which is the convention `src/metrics/stability.py` reads its
    before/update pairing from. Reversing it silently inverts EV's sign.

    `values`: [N, T, H, 1, Q]. `actual`/`mask`: [N, T, H, 1]. The trailing
    1 is the channel axis the stability metrics expect; every eval suite is
    univariate by the time it reaches here.
    """

    values: np.ndarray
    actual: np.ndarray
    mask: np.ndarray
    quantiles: list[float]
    stride: int


def run_rolling(
    forecaster: Forecaster,
    series: list[EvalSeries],
    horizon: int,
    stride: int,
    n_windows: int,
    batch_size: int = 256,
) -> RollingForecasts:
    """Re-forecasts each series from `n_windows` advancing creation dates.

    The suites' own protocols are all single-window or exactly
    non-overlapping, so EV and sFPC are undefined on them. This builds the
    overlapping windows those metrics need, which makes `stride` a declared
    parameter of this mode rather than anything inherited from a suite.

    Windows are emitted oldest-first and the last one ends at the series'
    final observation, so every window forecasts data the suite actually
    holds.
    """
    if stride >= horizon:
        raise ValueError(
            f"stride {stride} must be below the horizon {horizon}, otherwise "
            "windows do not overlap and forecast stability is undefined"
        )
    if horizon > forecaster.horizon:
        raise ValueError(
            f"rolling horizon {horizon} exceeds the model's native {forecaster.horizon}"
        )

    full = [np.concatenate([item.history, item.actual]) for item in series]
    needed = horizon + (n_windows - 1) * stride + 1
    too_short = [i for i, values in enumerate(full) if len(values) < needed]
    if too_short:
        raise ValueError(
            f"{len(too_short)} series hold fewer than the {needed} points "
            f"{n_windows} windows of {horizon} at stride {stride} require; "
            "lower n_windows rather than dropping them"
        )

    values, actuals, masks = [], [], []
    for window in range(n_windows):
        # window 0 is the oldest, so its horizon sits furthest from the end.
        back = (n_windows - 1 - window) * stride
        cut = [len(v) - horizon - back for v in full]
        sliced = [
            EvalSeries(
                suite=item.suite,
                subset=item.subset,
                item_id=item.item_id,
                history=v[:c],
                actual=v[c : c + horizon],
                period=item.period,
                freq=item.freq,
            )
            for item, v, c in zip(series, full, cut)
        ]
        out = run(forecaster, sliced, batch_size=batch_size)
        values.append(out.values)
        actuals.append(out.actual)
        masks.append(out.actual_mask)

    return RollingForecasts(
        # [N, T, H, Q] -> [N, T, H, C=1, Q]
        values=np.stack(values, axis=1)[:, :, :, None, :],
        actual=np.stack(actuals, axis=1)[..., None],
        mask=np.stack(masks, axis=1)[..., None],
        quantiles=list(forecaster.quantiles),
        stride=stride,
    )


def run(
    forecaster: Forecaster,
    series: list[EvalSeries],
    batch_size: int = 256,
) -> Forecasts:
    """Forecasts every series, in suite order.

    All series in one call must share a horizon, which every suite subset
    does by construction. The model predicts its native horizon and the
    result is truncated, which is valid only because every suite horizon is
    at most 60 against a native 128, so no autoregressive rollout is needed.
    """
    horizons = {len(item.actual) for item in series}
    if len(horizons) != 1:
        raise ValueError(
            f"mixed horizons {sorted(horizons)} in one call; group by subset first"
        )
    horizon = horizons.pop()
    if horizon > forecaster.horizon:
        raise ValueError(
            f"suite horizon {horizon} exceeds the model's native "
            f"{forecaster.horizon}; this would need autoregressive rollout, "
            "which the harness does not implement"
        )

    context, valid = build_context(series, forecaster.context_length)
    freqs = [item.freq for item in series]

    chunks = []
    for start in range(0, len(series), batch_size):
        stop = min(start + batch_size, len(series))
        predicted = forecaster.predict(
            context[start:stop], valid[start:stop], freqs[start:stop]
        )
        chunks.append(predicted[:, :horizon, :])
    values = np.concatenate(chunks, axis=0)

    if values.shape[0] != len(series):
        raise ValueError(
            f"{values.shape[0]} forecasts for {len(series)} series. Every "
            "series a suite defines must be scored; dropping short ones here "
            "would reinstate the context filter the suites deliberately drop."
        )

    actual = np.stack([item.actual for item in series])
    return Forecasts(
        values=values,
        actual=np.nan_to_num(actual, nan=0.0),
        actual_mask=np.isfinite(actual).astype(np.float64),
        history=context,
        history_mask=valid,
        periods=np.array([item.period for item in series]),
        quantiles=list(forecaster.quantiles),
        subsets=[item.subset for item in series],
        item_ids=[item.item_id for item in series],
    )

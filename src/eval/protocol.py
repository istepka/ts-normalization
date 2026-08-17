"""The interface `src/eval/predict.py` speaks to every model.

`src/eval/` holds no model-specific code. Each adapter implements
`build_forecaster` in its own module, next to the `build_*_model` /
`make_batch` / `forward` / `training_step_metrics` set those modules already
share, so the per-model padding and frequency-encoding quirks stay where the
rest of that model's quirks live.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# Forecast-only. MOMENT is built for masked reconstruction and has no
# forecast head, so it is out of scope for the harness by design.
FORECAST_MODELS = ("timesfm", "chronos2", "moirai2")


@runtime_checkable
class Forecaster(Protocol):
    """A loaded checkpoint that turns left-padded context into quantiles.

    `quantiles` are the levels of `predict`'s last axis and must contain 0.5,
    since the point metrics score the median. `horizon` is the model's native
    forecast length; callers ask for at most that many steps and truncate,
    which is sound only because every suite horizon is at most 60 against a
    native 128 (see notes/agentic_logs/2026-08-16-eval-harness.md).
    """

    quantiles: list[float]
    context_length: int
    horizon: int

    def predict(
        self, context: np.ndarray, valid: np.ndarray, freqs: list[str]
    ) -> np.ndarray:
        """Forecasts `horizon` steps in the series' original units.

        `context`: [N, context_length], left-padded, invalid positions zeroed.
        `valid`: [N, context_length], 1 = real observation. `freqs`: the
        pandas frequency alias per series. Returns [N, horizon, Q].
        """
        ...


def build_forecaster(cfg, checkpoint_path: Path, device: str) -> Forecaster:
    """Dispatches to the adapter module named by `cfg.model`.

    Imported lazily so loading the registry does not import every model's
    dependencies, and indexed rather than `.get`-ed so an unsupported model
    fails here with the supported list rather than deeper in.
    """
    from src.models import chronos2, moirai2, timesfm

    modules = {"timesfm": timesfm, "chronos2": chronos2, "moirai2": moirai2}
    if cfg.model not in modules:
        raise ValueError(
            f"model {cfg.model!r} has no forecaster; the harness scores "
            f"{FORECAST_MODELS} (MOMENT reconstructs rather than forecasts)"
        )
    return modules[cfg.model].build_forecaster(cfg, checkpoint_path, device)

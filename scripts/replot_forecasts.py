"""Regenerate the forecast-evolution figures from saved snapshot data.

Lets us restyle the qualitative figure without retraining: `main.py` dumps
`forecast_data_{label}.npz` per run, and this script reloads them and re-renders.

    uv run python scripts/replot_forecasts.py [output_dir]
"""

import sys
from pathlib import Path

import numpy as np

from src.plots import plot_forecast_evolution

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/loss_space_toy")

titles = {
    "normalized": "Forecast evolution — normalized-space loss",
    "original": "Forecast evolution — original-space loss",
}
for label, title in titles.items():
    data = np.load(out_dir / f"forecast_data_{label}.npz")
    history = {
        "forecast_steps": list(data["steps"]),
        "forecast_pred": list(data["pred"]),
        "probe_context": data["context"],
        "probe_target": data["target"],
    }
    names = list(data["names"])
    plot_forecast_evolution(
        history, names, title, out_dir / f"forecast_evolution_{label}.png"
    )
    print(f"replotted {label}")

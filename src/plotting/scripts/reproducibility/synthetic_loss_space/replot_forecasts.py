"""Regenerate the forecast-evolution figures from saved snapshot data.

Lets us restyle the qualitative figure without retraining: `main.py` dumps
`forecast_data_{label}.npz` per run, and this script reloads them and re-renders.

    uv run python -m src.plotting.scripts.reproducibility.synthetic_loss_space.replot_forecasts [output_dir]

Paper: regenerates the panels of Fig. loss_space_comp:synthetic:forecasts.
"""

import sys
from pathlib import Path

import numpy as np

from src.plotting.core.loss_space import plot_forecast_evolution

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/loss_space_toy")
paper_dir = out_dir.parent / f"{out_dir.name}_paper"
paper_dir.mkdir(parents=True, exist_ok=True)
columns = [0, 10, 50, 100, 500, 30000]

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
        history, names, title, columns, out_dir / f"forecast_evolution_{label}.png"
    )
    plot_forecast_evolution(
        history,
        names,
        title,
        columns,
        paper_dir / f"forecast_evolution_{label}.png",
        paper=True,
    )
    print(f"replotted {label}")

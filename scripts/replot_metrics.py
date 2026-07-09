"""Regenerate the nMSE convergence figures from saved per-step metric histories.

Lets us restyle/zoom the curves without retraining: `main.py` dumps
`metrics_{label}.npz` per run, and this script reloads them and re-renders the core,
controls, global, and gradient-magnitude figures (log + zoomed-linear).

    uv run python scripts/replot_metrics.py [output_dir] [zoom1,zoom2,...]
"""

import sys
from pathlib import Path

import numpy as np

from src.plots import plot_global_nmse, plot_grad_magnitude, plot_nmse_panels

SETUP_LABELS = ["normalized", "original", "original_equalvar", "original_gradmatch"]

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/loss_space_toy")
zoom_windows = [
    int(w)
    for w in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["200", "500", "2000"])
]
band = "std"

results, names = {}, None
for label in SETUP_LABELS:
    data = np.load(out_dir / f"metrics_{label}.npz")
    names = list(data["names"])
    step, nmse, gnmse, grad = (
        list(data["step"]),
        data["nmse"],  # [S, E, C]
        data["global_nmse"],  # [S, E]
        data["grad_mag"],  # [S, E, C]
    )
    histories = [
        {
            "step": step,
            "nmse": list(nmse[s]),
            "global_nmse": list(gnmse[s]),
            "grad_mag": list(grad[s]),
        }
        for s in range(nmse.shape[0])
    ]
    results[label] = {"histories": histories}

# Log figures span the full range; linear figures are saved once per zoom window.
variants = [("log", "", None)]
variants += [("linear", f"_linear_{w}", (0, w)) for w in zoom_windows]
for yscale, suffix, xlim in variants:
    plot_nmse_panels(
        results,
        names,
        ["normalized", "original"],
        ["Normalized-space loss", "Original-space loss"],
        band,
        out_dir / f"nmse_core{suffix}.png",
        yscale=yscale,
        xlim=xlim,
    )
    plot_nmse_panels(
        results,
        names,
        ["original_equalvar", "original_gradmatch"],
        ["Original + equal variance", "Original + grad-norm matched"],
        band,
        out_dir / f"nmse_controls{suffix}.png",
        yscale=yscale,
        xlim=xlim,
    )
    plot_global_nmse(
        results,
        SETUP_LABELS,
        band,
        out_dir / f"nmse_global{suffix}.png",
        yscale=yscale,
        xlim=xlim,
    )
plot_grad_magnitude(results, names, band, out_dir / "grad_magnitude.png")
plot_grad_magnitude(
    results, names, band, out_dir / "grad_magnitude_linear.png", yscale="linear"
)
print(f"replotted nMSE figures in {out_dir} (linear zooms {zoom_windows})")

"""Paper-ready captions, written next to each paper PDF as `{name}_caption.txt`.

Keyed by figure basename (matching the PDF filename without extension). Edit here to
keep the figure captions versioned with the code that produces them.
"""

from pathlib import Path

CAPTIONS = {
    "nmse_core": (
        "Per-category normalized MSE during training (in normalized space, the common "
        "metric), mean over 5 seeds with $\\pm 1$ SD bands; the dashed grey line is the "
        "global nMSE averaged over categories. Left: loss computed in the normalized "
        "space. Right: loss computed in the original space. Categories share the same "
        "mean but have variances $1\\!:\\!10\\!:\\!100$. Under original-space loss the "
        "per-category curves fan out by variance (var100 $\\ll$ var1) because the "
        "gradient is scaled by $b^2=\\sigma^2$, whereas normalized-space loss keeps them "
        "together. All curves descend throughout training, so the disparity is a "
        "difference in convergence \\emph{rate}, not in asymptote."
    ),
    "nmse_controls": (
        "Control experiments, per-category nMSE (mean $\\pm 1$ SD over 5 seeds; dashed "
        "grey = global nMSE). Left: equal-variance data (all categories rescaled to "
        "variance 1, distinct frequencies kept) -- the original-space fan-out collapses "
        "to the normalized baseline, confirming the disparity is driven by variance and "
        "not by per-category task difficulty. Right: gradient-norm-matched original-space "
        "training (the full gradient is rescaled to unit global norm each step) -- the "
        "fan-out persists, showing the effect is loss-space bias and not a global "
        "learning-rate artifact."
    ),
    "nmse_global": (
        "Global nMSE (averaged over all categories) during training for each setup, mean "
        "$\\pm 1$ SD over 5 seeds. All setups converge; original-space loss attains the "
        "lowest global error fastest because high-variance categories receive amplified "
        "gradients, while the gradient-norm-matched control is slowest under unit-norm "
        "steps."
    ),
    "grad_magnitude": (
        "Per-category gradient magnitude $\\|\\partial\\mathcal{L}/\\partial\\hat z\\|$ at "
        "initialization (mean $\\pm 1$ SD over 5 seeds). Under normalized-space loss the "
        "magnitudes are category-independent; under original-space loss they scale as "
        "$b^2=\\sigma^2$ ($\\approx 1\\!:\\!10\\!:\\!100$), the direct signature of the "
        "variance-induced gradient bias."
    ),
    "forecast_evolution_original": (
        "Qualitative forecast evolution under original-space loss. Rows are categories "
        "(variance increasing top to bottom), columns are selected training steps; each "
        "cell shows the target (black) and model prediction (red) over the forecast "
        "horizon. High-variance categories match the "
        "target within a few hundred steps while the low-variance category lags by orders "
        "of magnitude -- the disparate convergence rate induced by the $\\sigma^2$ "
        "gradient scaling."
    ),
    "forecast_evolution_normalized": (
        "Qualitative forecast evolution under normalized-space loss (rows: categories, "
        "columns: selected training steps; black target, red prediction). With the "
        "variance factor removed from the gradient, all "
        "categories converge on a similar timescale, in contrast to the original-space "
        "case."
    ),
    "forecasts_original": (
        "Final-state forecasts under original-space loss: the trained model fits the "
        "high-variance categories (var10, var100) near-perfectly while under-serving the "
        "low-variance category (var1), which received the smallest gradients."
    ),
    "forecasts_normalized": (
        "Final-state forecasts under normalized-space loss: with category-independent "
        "gradients the model serves all three categories evenly."
    ),
}


def write_captions(paper_dir: Path):
    for name, text in CAPTIONS.items():
        (paper_dir / f"{name}_caption.txt").write_text(text + "\n")

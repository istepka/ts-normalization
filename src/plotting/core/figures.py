"""Figure-level helpers shared by the paper figure scripts."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

GLOBAL_STYLE = {"color": "0.35", "linestyle": "--", "linewidth": 1.6}

PAPER_RCPARAMS = {
    # Embed TrueType rather than Type 3 fonts. Type 3 is what matplotlib emits
    # by default and what most venues reject at submission.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Keep SVG text as text so it stays selectable and searchable.
    "svg.fonttype": "none",
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.02,
    # Opaque background. Transparent figures read as invisible text on the
    # dark backgrounds some PDF viewers and slide decks use.
    "savefig.transparent": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    # Hairlines disappear in print. These are the thinnest safe widths.
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
}


def apply_paper_style() -> None:
    """Install the paper defaults, most importantly TrueType font embedding."""
    mpl.rcParams.update(PAPER_RCPARAMS)


def mean_ci(
    values: np.ndarray, axis: int = 0, log_space: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and 95% t-interval across `axis`, optionally in log10 space."""
    work = np.log10(values) if log_space else np.asarray(values)
    count = work.shape[axis]
    mean = work.mean(axis=axis)
    half = t.ppf(0.975, count - 1) * work.std(axis=axis, ddof=1) / np.sqrt(count)
    if log_space:
        return 10**mean, 10 ** (mean - half), 10 ** (mean + half)
    return mean, mean - half, mean + half


def save_figure(
    fig: plt.Figure,
    base_path: Path,
    dpi: int = 300,
    tight_layout: bool = False,
    rect: tuple[float, float, float, float] | None = None,
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.02,
) -> None:
    """Write `base_path` as both .pdf (for the paper) and .png (for previews).

    `bbox_inches="tight"` crops the whitespace LaTeX would otherwise have to
    absorb. Pass `bbox_inches=None` for panels that are typeset side by side as
    subfigures, where cropping each one separately would leave their axes
    misaligned. The PDF carries no creation date, so re-running a figure script
    without changing the data leaves the file byte identical.
    """
    if tight_layout:
        fig.tight_layout(rect=rect)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    common = {"bbox_inches": bbox_inches, "pad_inches": pad_inches}
    fig.savefig(
        base_path.with_suffix(".pdf"), metadata={"CreationDate": None}, **common
    )
    fig.savefig(base_path.with_suffix(".png"), dpi=dpi, **common)
    plt.close(fig)


def style_step_axis(axis: plt.Axes, end_step: int = 30_000) -> None:
    """Apply the shared training-step axis styling used by convergence panels."""
    ticks = np.arange(0, end_step + 1, 10_000)
    axis.set_xlim(0, end_step)
    axis.set_xticks(
        ticks, labels=[f"{step // 1_000}k" if step else "0" for step in ticks]
    )
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.tick_params(labelsize=6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def bottom_legend(fig: plt.Figure, handles, labels, ncol: int) -> None:
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        ncol=ncol,
        fontsize=7,
    )

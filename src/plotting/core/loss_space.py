"""Figures for the loss-space comparison toy.

All learning curves report nMSE in normalized space (the common metric), so
normalized-space and original-space runs are directly comparable. Every figure can
be rendered in a `paper` style (PDF only, no titles, minimal margins, deduplicated
legends/labels) for inclusion in the paper.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

from src.plotting.core.figures import GLOBAL_STYLE, apply_paper_style
from src.plotting.core.palette import PRIMARY, TEAL, apply_palette

apply_palette()
apply_paper_style()

DISPLAY_LABELS = {
    "normalized": "Normalized-space",
    "original": "Original-space",
    "original_equalvar": "Equal variance",
    "original_gradmatch": "Gradient-norm matched",
}


def _hide_top_right_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, out_path: Path, paper: bool = False):
    """Paper: PDF only with tight margins. Otherwise PNG (preview) + PDF."""
    metadata = {"CreationDate": None}
    if paper:
        fig.savefig(
            out_path.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=0.02,
            metadata=metadata,
        )
    else:
        fig.savefig(out_path, dpi=150)
        fig.savefig(out_path.with_suffix(".pdf"), metadata=metadata)
    plt.close(fig)


def _mean_band(histories: list[dict], key: str, band: str, space: str = "log"):
    """Stack a per-step metric across seeds; return (steps, center, lower, upper).

    `space` matches the axis the figure is drawn on, so the aggregation is honest:
    - "log": geometric mean +/- log-space std. Needed on a log axis -- a linear
      mean +/- std would dip below zero whenever std >= mean (which Adam's near-floor
      noise routinely produces), and matplotlib clips that to the axis floor and
      paints a misleadingly huge fill. Symmetric and positive in log space.
    - "linear": ordinary arithmetic mean +/- std. The truthful absolute-scale view --
      use it on linear axes so tiny near-floor fluctuations are not log-amplified into
      large-looking swings.

    `band` selects the halfwidth: "std" = 1 SD across seeds, "se" = std / sqrt(n)."""
    steps = histories[0]["step"]
    arr = np.array([np.stack(h[key], axis=0) for h in histories])  # [S, E, ...]
    work = np.log10(arr) if space == "log" else arr
    mean = work.mean(axis=0)
    std = work.std(axis=0, ddof=1)
    if band == "se":
        std = std / np.sqrt(arr.shape[0])
    elif band != "std":
        raise ValueError(f"unknown band: {band}")
    if space == "log":
        return steps, 10**mean, 10 ** (mean - std), 10 ** (mean + std)
    if space == "linear":
        return steps, mean, mean - std, mean + std
    raise ValueError(f"unknown space: {space}")


def _nmse_curve(
    ax, histories, names, band, *, title, show_ylabel, show_legend, yscale, xlim
):
    space = "linear" if yscale == "linear" else "log"
    steps, mean, lo, hi = _mean_band(
        histories, "nmse", band, space
    )  # [n_eval, num_cat]
    for c, name in enumerate(names):
        (line,) = ax.plot(steps, mean[:, c], label=name)
        ax.fill_between(
            steps,
            lo[:, c],
            hi[:, c],
            color=line.get_color(),
            alpha=0.25,
            linewidth=0,
        )
    gsteps, gmean, _, _ = _mean_band(histories, "global_nmse", band, space)
    ax.plot(gsteps, gmean, label="global", **GLOBAL_STYLE)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel("step")
    if show_ylabel:
        ax.set_ylabel("nMSE")
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend()
    _hide_top_right_spines(ax)


def plot_nmse_panels(
    results, names, labels, titles, band, out_path, paper=False, yscale="log", xlim=None
):
    fig, axes = plt.subplots(
        1, len(labels), figsize=(6 * len(labels), 4.5), sharey=True
    )
    if len(labels) == 1:
        axes = [axes]
    for j, (ax, label, title) in enumerate(zip(axes, labels, titles)):
        _nmse_curve(
            ax,
            results[label]["histories"],
            names,
            band,
            title=None if paper else title,
            show_ylabel=(j == 0),
            show_legend=(j == 0),
            yscale=yscale,
            xlim=xlim,
        )
    fig.tight_layout()
    _save(fig, out_path, paper)


def plot_nmse_subfigures(results, names, band, out_dir, paper=False):
    """Compact core-and-control panels plus a separate shared legend."""
    panels = (
        ("normalized", "normalized", (0, 500), True, False),
        ("original", "original", (0, 500), False, False),
        ("original_equalvar", "equalvar", (0, 2000), True, True),
        (
            "original_gradmatch",
            "gradmatch",
            (0, 2000),
            False,
            True,
        ),
    )
    legend_handles = None
    legend_labels = None
    for label, filename, xlim, show_ylabel, show_xlabel in panels:
        fig, ax = plt.subplots(figsize=(3.8, 1.75))
        _nmse_curve(
            ax,
            results[label]["histories"],
            names,
            band,
            title=None,
            show_ylabel=show_ylabel,
            show_legend=False,
            yscale="linear",
            xlim=xlim,
        )
        if not show_ylabel:
            ax.tick_params(axis="y", labelleft=False)
        if not show_xlabel:
            ax.set_xlabel("")
        legend_handles, legend_labels = ax.get_legend_handles_labels()
        fig.tight_layout()
        _save(fig, out_dir / f"nmse_{filename}.png", paper)

    fig = plt.figure(figsize=(6.0, 0.32))
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center",
        ncol=4,
        frameon=False,
    )
    _save(fig, out_dir / "nmse_shared_legend.png", paper)


def plot_global_nmse(
    results,
    labels,
    band,
    out_path,
    paper=False,
    yscale="log",
    xlim=None,
    show_legend=True,
):
    """Global nMSE (averaged over all categories) vs step, mean +/- 1 band per setup."""
    space = "linear" if yscale == "linear" else "log"
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for label in labels:
        steps, mean, lo, hi = _mean_band(
            results[label]["histories"], "global_nmse", band, space
        )
        (line,) = ax.plot(steps, mean, label=DISPLAY_LABELS[label])
        ax.fill_between(
            steps,
            lo,
            hi,
            color=line.get_color(),
            alpha=0.25,
            linewidth=0,
        )
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel("step")
    ax.set_ylabel("global nMSE")
    if not paper:
        ax.set_title("Global convergence")
    if show_legend:
        fig.legend(
            *ax.get_legend_handles_labels(),
            loc="center right",
            bbox_to_anchor=(0.99, 0.5),
            ncol=1,
            frameon=False,
        )
    _hide_top_right_spines(ax)
    right = 0.68 if show_legend else 1.0
    fig.tight_layout(rect=(0.0, 0.0, right, 1.0))
    _save(fig, out_path, paper)


def plot_grad_magnitude(
    results,
    names,
    band,
    out_path,
    paper=False,
    yscale="log",
    show_legend=True,
):
    """Per-category gradient magnitude at init (step 0), where the b^2 = sigma^2
    scaling of the original-space gradient is exact. Bars are mean +/- 1 band."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(names))
    labels = (
        "normalized",
        "original",
        "original_equalvar",
        "original_gradmatch",
    )
    width = 0.18
    offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2) * width
    for offset, label in zip(offsets, labels):
        grads = np.array([h["grad_mag"][0] for h in results[label]["histories"]])
        mean = grads.mean(axis=0)
        std = grads.std(axis=0, ddof=1)
        half = std if band == "std" else std / np.sqrt(grads.shape[0])
        ax.bar(
            x + offset,
            mean,
            width,
            yerr=half,
            capsize=3,
            label=DISPLAY_LABELS[label],
        )
    ax.set_yscale(yscale)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlabel("category (increasing variance)")
    ax.set_ylabel(r"$\|\partial \mathcal{L}/\partial \hat z\|$")
    if not paper:
        ax.set_title("Per-category gradient magnitude near init")
    if show_legend:
        fig.legend(
            *ax.get_legend_handles_labels(),
            loc="center right",
            bbox_to_anchor=(0.99, 0.5),
            ncol=1,
            frameon=False,
        )
    _hide_top_right_spines(ax)
    right = 0.67 if show_legend else 1.0
    fig.tight_layout(rect=(0.0, 0.0, right, 1.0))
    _save(fig, out_path, paper)


def plot_setup_legend(out_path, paper=False):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    handles = [
        Line2D([0], [0], color=colors[index], linewidth=2)
        for index in range(len(DISPLAY_LABELS))
    ]
    fig = plt.figure(figsize=(6.5, 0.35))
    fig.legend(
        handles,
        DISPLAY_LABELS.values(),
        loc="center",
        ncol=4,
        frameon=False,
    )
    _save(fig, out_path, paper)


def _fmt_step(step: int) -> str:
    return f"{step // 1000}k" if step >= 1000 else str(step)


def _probe_geometry(history):
    ctx = np.array(history["probe_context"])  # [num_categories, L]
    tgt = np.array(history["probe_target"])  # [num_categories, horizon]
    num_categories, ctx_len = ctx.shape
    horizon = tgt.shape[1]
    tail = min(ctx_len, 2 * horizon)
    t_tail = np.arange(ctx_len - tail, ctx_len)
    t_tgt = np.arange(ctx_len, ctx_len + horizon)
    return ctx, tgt, num_categories, ctx_len, tail, t_tail, t_tgt


def plot_forecast_evolution(history, names, title, columns, out_path, paper=False):
    """Small-multiples grid: rows = categories, columns = the requested training steps
    (each mapped to the nearest captured snapshot). Each cell zooms into the forecast
    (short context tail + horizon) with target (black) and prediction (red). Reading
    DOWN a column at an early step shows the disparate rate; ACROSS a row, convergence."""
    pred_by_step = {
        int(s): p for s, p in zip(history["forecast_steps"], history["forecast_pred"])
    }
    available = sorted(pred_by_step)
    cols, seen = [], set()
    for c in columns:
        nearest = min(available, key=lambda s: abs(s - int(c)))
        if nearest not in seen:
            seen.add(nearest)
            cols.append((int(c), nearest))  # (requested label, captured step)
    ctx, tgt, num_categories, ctx_len, tail, t_tail, t_tgt = _probe_geometry(history)

    fig, axes = plt.subplots(
        num_categories,
        len(cols),
        figsize=(2.3 * len(cols), 2.1 * num_categories),
        squeeze=False,
    )
    for r, name in enumerate(names):
        lo = min(ctx[r, -tail:].min(), tgt[r].min())
        hi = max(ctx[r, -tail:].max(), tgt[r].max())
        pad = 0.12 * (hi - lo + 1e-9)
        for j, (label_step, step) in enumerate(cols):
            ax = axes[r][j]
            ax.plot(t_tail, ctx[r, -tail:], color="0.75", lw=1)
            ax.plot(t_tgt, tgt[r], color="black", lw=2)
            ax.plot(t_tgt, pred_by_step[step][r], color=PRIMARY, lw=1.6)
            ax.axvline(ctx_len - 0.5, color="0.85", lw=0.8)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xticks([])
            ax.set_yticks([])
            _hide_top_right_spines(ax)
            if r == 0:
                ax.set_title(
                    f"step {_fmt_step(label_step)}", fontsize=15, fontweight="bold"
                )
            if j == 0:
                ax.set_ylabel(name, fontsize=15, fontweight="bold")
    if not paper:
        fig.suptitle(title)
    fig.tight_layout()
    _save(fig, out_path, paper)


def plot_qualitative(cfg, dataset, model, out_path, paper=False):
    model.eval()
    fig, axes = plt.subplots(
        1, dataset.num_categories, figsize=(5 * dataset.num_categories, 4)
    )
    if dataset.num_categories == 1:
        axes = [axes]
    ctx_len = dataset.context_length
    with torch.no_grad():
        for c, (ax, name, windows) in enumerate(
            zip(axes, dataset.category_names, dataset.windows)
        ):
            window = windows[0:1].to(cfg.device)
            context = window[:, :ctx_len]
            target = window[:, ctx_len:]
            z_pred, a, b = model(context)
            y_pred = (b * z_pred + a).cpu().numpy().ravel()
            ctx = context.cpu().numpy().ravel()
            tgt = target.cpu().numpy().ravel()
            t_ctx = np.arange(ctx_len)
            t_tgt = np.arange(ctx_len, ctx_len + len(tgt))
            ax.plot(t_ctx, ctx, color="black", label="context")
            ax.plot(t_tgt, tgt, color=TEAL, label="target")
            ax.plot(t_tgt, y_pred, color=PRIMARY, ls="--", label="prediction")
            ax.set_title(name)
            if c == 0:
                ax.legend()
    fig.tight_layout()
    _save(fig, out_path, paper)


def gif_forecast_evolution(history, names, out_path, fps=2):
    """Animated forecast: the prediction (red) morphs toward the target (black) as
    training proceeds, one panel per category. High-variance categories lock on fast,
    the low-variance one lags — the disparate convergence rate, animated."""
    steps = sorted(int(s) for s in history["forecast_steps"])
    pred_by_step = {
        int(s): p for s, p in zip(history["forecast_steps"], history["forecast_pred"])
    }
    ctx, tgt, num_categories, ctx_len, tail, t_tail, t_tgt = _probe_geometry(history)

    fig, axes = plt.subplots(1, num_categories, figsize=(4 * num_categories, 3.4))
    if num_categories == 1:
        axes = [axes]
    pred_lines = []
    for r, (ax, name) in enumerate(zip(axes, names)):
        lo = min(ctx[r, -tail:].min(), tgt[r].min())
        hi = max(ctx[r, -tail:].max(), tgt[r].max())
        pad = 0.12 * (hi - lo + 1e-9)
        ax.plot(t_tail, ctx[r, -tail:], color="0.75", lw=1)
        ax.plot(t_tgt, tgt[r], color="black", lw=2, label="target")
        (lp,) = ax.plot(
            t_tgt,
            pred_by_step[steps[0]][r],
            color=PRIMARY,
            lw=1.8,
            label="prediction",
        )
        ax.axvline(ctx_len - 0.5, color="0.85", lw=0.8)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
        pred_lines.append(lp)
    # Horizontal legend below the axes so it never occludes the curves.
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    step_text = fig.text(0.5, 0.965, f"step {steps[0]}", ha="center", fontsize=12)

    def update(i):
        s = steps[i]
        for r, lp in enumerate(pred_lines):
            lp.set_ydata(pred_by_step[s][r])
        step_text.set_text(f"step {s}")
        return pred_lines

    anim = FuncAnimation(fig, update, frames=len(steps), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def gif_nmse_convergence(results, names, band, out_path, n_frames=45, fps=8):
    """Animated convergence: the per-category nMSE curves (and global, dashed) draw in
    progressively for normalized- vs original-space loss, so the fan-out emerges live."""
    labels = ["normalized", "original"]
    data = {}
    for label in labels:
        steps, mean, _, _ = _mean_band(results[label]["histories"], "nmse", band)
        _, gmean, _, _ = _mean_band(results[label]["histories"], "global_nmse", band)
        data[label] = (np.array(steps), mean, gmean)
    n_eval = data["normalized"][1].shape[0]
    frames = np.unique(np.linspace(2, n_eval, n_frames).astype(int))
    lo = min(d[1].min() for d in data.values())
    hi = max(d[1].max() for d in data.values())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    artists = {}
    for ax, label in zip(axes, labels):
        ax.set_yscale("log")
        ax.set_xlim(0, data[label][0][-1])
        ax.set_ylim(lo * 0.5, hi * 2)
        ax.set_xlabel("step")
        ax.set_title(label)
        per = [ax.plot([], [], label=n)[0] for n in names]
        (gl,) = ax.plot([], [], label="global", **GLOBAL_STYLE)
        artists[label] = (per, gl)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("nMSE")

    def update(k):
        i = frames[k]
        out = []
        for label in labels:
            steps, mean, gmean = data[label]
            per, gl = artists[label]
            for c, line in enumerate(per):
                line.set_data(steps[:i], mean[:i, c])
                out.append(line)
            gl.set_data(steps[:i], gmean[:i])
            out.append(gl)
        return out

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

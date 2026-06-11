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


GLOBAL_STYLE = {"color": "0.35", "linestyle": "--", "linewidth": 1.6}


def _save(fig, out_path: Path, paper: bool = False):
    """Paper: PDF only with tight margins. Otherwise PNG (preview) + PDF."""
    if paper:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(out_path, dpi=150)
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _mean_band(histories: list[dict], key: str, band: str):
    """Stack a per-step metric across seeds; return (steps, center, lower, upper).

    Statistics are computed in log10 space because every consumer plots on a log
    y-axis and the per-seed nMSE spans orders of magnitude. A linear mean +/- std
    band puts its lower edge below zero whenever std >= mean (which Adam's noisy
    trajectories routinely produce); on a log axis matplotlib then clips that edge
    to the axis floor and paints a misleadingly huge fill. center is the geometric
    mean; the band is +/- one log-space std ("std") or std/sqrt(n) ("se"), so it is
    always positive and symmetric in the space that is actually displayed.

    `band` selects the halfwidth: "std" = 1 standard deviation across seeds,
    "se" = std / sqrt(n_seeds)."""
    steps = histories[0]["step"]
    arr = np.array([np.stack(h[key], axis=0) for h in histories])  # [S, E, ...]
    log = np.log10(arr)
    mean = log.mean(axis=0)
    std = log.std(axis=0, ddof=1)
    if band == "se":
        std = std / np.sqrt(arr.shape[0])
    elif band != "std":
        raise ValueError(f"unknown band: {band}")
    return steps, 10**mean, 10 ** (mean - std), 10 ** (mean + std)


def _nmse_curve(ax, histories, names, band, *, title, show_ylabel, show_legend, yscale):
    steps, mean, lo, hi = _mean_band(histories, "nmse", band)  # [n_eval, num_cat]
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
    gsteps, gmean, _, _ = _mean_band(histories, "global_nmse", band)
    ax.plot(gsteps, gmean, label="global", **GLOBAL_STYLE)
    ax.set_yscale(yscale)
    ax.set_xlabel("step")
    if show_ylabel:
        ax.set_ylabel("nMSE")
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend()


def plot_nmse_panels(
    results, names, labels, titles, band, out_path, paper=False, yscale="log"
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
        )
    fig.tight_layout()
    _save(fig, out_path, paper)


def plot_global_nmse(results, labels, band, out_path, paper=False, yscale="log"):
    """Global nMSE (averaged over all categories) vs step, mean +/- 1 band per setup."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label in labels:
        steps, mean, lo, hi = _mean_band(
            results[label]["histories"], "global_nmse", band
        )
        (line,) = ax.plot(steps, mean, label=label)
        ax.fill_between(
            steps,
            lo,
            hi,
            color=line.get_color(),
            alpha=0.25,
            linewidth=0,
        )
    ax.set_yscale(yscale)
    ax.set_xlabel("step")
    ax.set_ylabel("global nMSE")
    if not paper:
        ax.set_title("Global convergence")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path, paper)


def plot_grad_magnitude(results, names, band, out_path, paper=False):
    """Per-category gradient magnitude at init (step 0), where the b^2 = sigma^2
    scaling of the original-space gradient is exact. Bars are mean +/- 1 band."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(names))
    width = 0.38
    for offset, label in zip((-width / 2, width / 2), ("normalized", "original")):
        grads = np.array([h["grad_mag"][0] for h in results[label]["histories"]])
        mean = grads.mean(axis=0)
        std = grads.std(axis=0, ddof=1)
        half = std if band == "std" else std / np.sqrt(grads.shape[0])
        ax.bar(x + offset, mean, width, yerr=half, capsize=3, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlabel("category (increasing variance)")
    ax.set_ylabel(r"$\|\partial \mathcal{L}/\partial \hat z\|$")
    if not paper:
        ax.set_title("Per-category gradient magnitude near init")
    ax.legend()
    fig.tight_layout()
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
            ax.plot(t_tgt, pred_by_step[step][r], color="tab:red", lw=1.6)
            ax.axvline(ctx_len - 0.5, color="0.85", lw=0.8)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xticks([])
            ax.set_yticks([])
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
            ax.plot(t_tgt, tgt, color="tab:green", label="target")
            ax.plot(t_tgt, y_pred, color="tab:red", ls="--", label="prediction")
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
            color="tab:red",
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

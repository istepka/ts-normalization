"""Figures for the loss-space comparison toy.

All learning curves report per-category nMSE in normalized space (the common metric),
so normalized-space and original-space runs are directly comparable.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import SyntheticTSDataset
from src.model import PatchTransformer


def _save(fig, out_path: Path):
    """Save both a PNG (notes/preview) and a PDF (paper) of the figure."""
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _mean_band(histories: list[dict], key: str, band: str):
    """Stack a per-step metric across seeds and return (steps, mean, halfwidth).

    `band` selects the shaded halfwidth: "std" = 1 standard deviation across
    seeds, "se" = std / sqrt(n_seeds). For per-category metrics each entry is
    [num_categories]; for scalar metrics the trailing axis is absent."""
    steps = histories[0]["step"]
    arr = np.array([np.stack(h[key], axis=0) for h in histories])  # [S, E, ...]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1)
    if band == "std":
        return steps, mean, std
    if band == "se":
        return steps, mean, std / np.sqrt(arr.shape[0])
    raise ValueError(f"unknown band: {band}")


def _nmse_curve(ax, histories: list[dict], names: list[str], title: str, band: str):
    steps, mean, half = _mean_band(histories, "nmse", band)  # [n_eval, num_cat]
    for c, name in enumerate(names):
        (line,) = ax.plot(steps, mean[:, c], label=name)
        ax.fill_between(
            steps,
            mean[:, c] - half[:, c],
            mean[:, c] + half[:, c],
            color=line.get_color(),
            alpha=0.25,
            linewidth=0,
        )
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("per-category nMSE (normalized space)")
    ax.set_title(title)
    ax.legend()


def plot_nmse_panels(results, names, labels, titles, band, out_path: Path):
    fig, axes = plt.subplots(
        1, len(labels), figsize=(6 * len(labels), 4.5), sharey=True
    )
    if len(labels) == 1:
        axes = [axes]
    for ax, label, title in zip(axes, labels, titles):
        _nmse_curve(ax, results[label]["histories"], names, title, band)
    fig.tight_layout()
    _save(fig, out_path)


def plot_global_nmse(results, labels, band, out_path: Path):
    """Global nMSE (averaged over all categories) vs step, mean +/- 1 band across
    seeds, to show overall convergence per setup."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label in labels:
        steps, mean, half = _mean_band(results[label]["histories"], "global_nmse", band)
        (line,) = ax.plot(steps, mean, label=label)
        ax.fill_between(
            steps,
            mean - half,
            mean + half,
            color=line.get_color(),
            alpha=0.25,
            linewidth=0,
        )
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("global nMSE (normalized space)")
    ax.set_title("Global convergence")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)


def plot_grad_magnitude(results, names, band, out_path: Path):
    """Per-category gradient magnitude at init (step 0), where the b^2 = sigma^2
    scaling of the original-space gradient is exact (later evals attenuate it as
    high-variance categories learn fast). Bars are mean +/- 1 band across seeds."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(names))
    width = 0.38
    for offset, label in zip((-width / 2, width / 2), ("normalized", "original")):
        grads = np.array(
            [h["grad_mag"][0] for h in results[label]["histories"]]
        )  # [S, num_categories]
        mean = grads.mean(axis=0)
        std = grads.std(axis=0, ddof=1)
        half = std if band == "std" else std / np.sqrt(grads.shape[0])
        ax.bar(x + offset, mean, width, yerr=half, capsize=3, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlabel("category (increasing variance)")
    ax.set_ylabel(r"$\|\partial \mathcal{L}/\partial \hat z\|$ (mean over rows)")
    ax.set_title("Per-category gradient magnitude near init")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)


def plot_forecast_evolution(history, names, title, out_path: Path):
    """Probe forecast at each snapshot step, colored light->dark by training step,
    one panel per category. Visualizes the disparate convergence rate: high-variance
    categories match the target after few steps, low-variance ones catch up late."""
    steps = history["forecast_steps"]
    preds = np.array(history["forecast_pred"])  # [n_snap, num_categories, horizon]
    ctx = np.array(history["probe_context"])  # [num_categories, L]
    tgt = np.array(history["probe_target"])  # [num_categories, horizon]
    num_categories, ctx_len = ctx.shape
    horizon = tgt.shape[1]
    t_ctx = np.arange(ctx_len)
    t_tgt = np.arange(ctx_len, ctx_len + horizon)

    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=min(steps), vmax=max(steps))
    fig, axes = plt.subplots(1, num_categories, figsize=(5 * num_categories, 4))
    if num_categories == 1:
        axes = [axes]
    for c, (ax, name) in enumerate(zip(axes, names)):
        ax.plot(t_ctx, ctx[c], color="0.6", lw=1)
        ax.plot(t_tgt, tgt[c], color="black", lw=2.2, label="target")
        for s, pred in zip(steps, preds):
            ax.plot(t_tgt, pred[c], color=cmap(norm(s)), lw=1.0)
        ax.set_title(name)
        ax.set_xlabel("t")
        ax.legend(loc="upper left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="training step")
    fig.suptitle(title)
    _save(fig, out_path)


def plot_qualitative(
    cfg, dataset: SyntheticTSDataset, model: PatchTransformer, out_path
):
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
            ax.set_title(f"{name}")
            ax.legend()
    fig.tight_layout()
    _save(fig, out_path)

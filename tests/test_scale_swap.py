import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from scripts.aggregate_scale_swap import (
    build_summary,
    early_log_auc,
    load_assignment,
    plot_curves,
    plot_gradient_ratios,
    plot_paired_auc,
    validate_assignments,
)
from src.data import RealScaleSwapDataset
from src.train import Trainer


def scale_swap_cfg(paths, scales):
    return OmegaConf.create(
        {
            "data": {
                "context_length": 4,
                "horizon": 2,
                "real_sources": [
                    {"name": f"source_{i}", "path": str(path), "key": "data"}
                    for i, path in enumerate(paths)
                ],
                "scale_assignment": scales,
                "real_shape_path": str(paths[0]),
                "real_shape_key": "data",
                "real_shape_val_fraction": 0.4,
                "real_value_scale": 1.0,
                "val_windows_per_category": 2,
            },
            "train": {"batch_size": 4, "steps": 2},
        }
    )


def write_sources(tmp_path):
    paths = []
    time = np.arange(10, dtype=np.float64)[None, :]
    row_offsets = 100.0 * np.arange(5, dtype=np.float64)[:, None]
    for source, signal in enumerate((time, time**2 + 0.5 * time)):
        path = tmp_path / f"source_{source}.npz"
        np.savez(path, data=row_offsets + signal)
        paths.append(path)
    return paths


def test_scale_swap_normalizes_each_source_before_scaling(tmp_path):
    dataset = RealScaleSwapDataset(
        scale_swap_cfg(write_sources(tmp_path), [1.0, 10.0]),
        torch.Generator().manual_seed(7),
    )

    for windows, scale in zip(dataset.windows, (1.0, 10.0)):
        context = windows[:, : dataset.context_length]
        torch.testing.assert_close(context.mean(dim=1), torch.zeros(len(context)))
        torch.testing.assert_close(
            context.std(dim=1, correction=0),
            torch.full((len(context),), scale),
        )


def test_swapped_assignments_keep_identical_sample_schedule(tmp_path):
    paths = write_sources(tmp_path)
    assignment_a = RealScaleSwapDataset(
        scale_swap_cfg(paths, [1.0, 10.0]),
        torch.Generator().manual_seed(11),
    )
    assignment_b = RealScaleSwapDataset(
        scale_swap_cfg(paths, [10.0, 1.0]),
        torch.Generator().manual_seed(11),
    )

    for batch_a, batch_b in zip(
        assignment_a.batch_schedule, assignment_b.batch_schedule
    ):
        torch.testing.assert_close(batch_a.category, batch_b.category)
        ratios = torch.where(batch_a.category == 0, 10.0, 0.1).unsqueeze(1)
        torch.testing.assert_close(batch_b.context, ratios * batch_a.context)
        torch.testing.assert_close(batch_b.target, ratios * batch_a.target)


def test_early_log_auc_is_lower_for_faster_curve():
    steps = np.array([0, 1000, 2000, 3000])
    slow = np.array([[[1.0, 0.8, 0.6, 0.5]], [[1.0, 0.9, 0.7, 0.6]]])
    fast = np.array([[[1.0, 0.3, 0.1, 0.05]], [[1.0, 0.4, 0.2, 0.1]]])

    assert np.all(early_log_auc(fast, steps) < early_log_auc(slow, steps))


def test_trainer_rejects_non_finite_metrics():
    with pytest.raises(FloatingPointError, match="validation nMSE at step 10"):
        Trainer._require_finite(np.array([1.0, np.nan]), "validation nMSE", 10)


def test_scale_swap_aggregation_writes_three_figures(tmp_path):
    names = np.array([f"dataset_{i}" for i in range(8)])
    steps = np.array([0, 1000, 2000])
    assignment_a = np.array([1.0] * 4 + [10.0] * 4)
    assignment_b = np.array([10.0] * 4 + [1.0] * 4)
    paths = []
    for label, assignment in (("a", assignment_a), ("b", assignment_b)):
        path = tmp_path / label
        path.mkdir()
        paths.append(path)
        for mode in ("normalized", "original"):
            nmse = np.empty((2, len(steps), len(names)))
            grad_mag = np.empty_like(nmse)
            for category, scale in enumerate(assignment):
                if mode == "original" and scale == 10.0:
                    nmse[:, :, category] = [1.0, 0.3, 0.1]
                else:
                    nmse[:, :, category] = [1.0, 0.8, 0.6]
                grad_mag[:, :, category] = scale**2 if mode == "original" else 1.0
            np.savez(
                path / f"metrics_{mode}.npz",
                names=names,
                step=steps,
                nmse=nmse,
                grad_mag=grad_mag,
            )

    a = load_assignment(paths[0])
    b = load_assignment(paths[1])
    validate_assignments(a, b)
    plot_curves(a, b, tmp_path / "curves.png")
    plot_paired_auc(a, b, tmp_path / "auc.png")
    plot_gradient_ratios(a, b, tmp_path / "gradients.png")
    summary = build_summary(a, b)

    assert (tmp_path / "curves.png").exists()
    assert (tmp_path / "auc.png").exists()
    assert (tmp_path / "gradients.png").exists()
    assert (
        summary["modes"]["original"]["paired_auc_low_minus_high"]["mean_ci95"]["mean"]
        > 0.0
    )


def test_scale_swap_aggregation_rejects_non_finite_metrics(tmp_path):
    names = np.array([f"dataset_{i}" for i in range(8)])
    nmse = np.ones((1, 3, 8))
    nmse[0, 1, 0] = np.nan
    np.savez(
        tmp_path / "metrics_normalized.npz",
        names=names,
        step=np.array([0, 1000, 2000]),
        nmse=nmse,
        grad_mag=np.ones_like(nmse),
    )

    with pytest.raises(ValueError, match="non-finite normalized nmse"):
        load_assignment(tmp_path)

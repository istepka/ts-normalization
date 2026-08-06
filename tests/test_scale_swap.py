import numpy as np
import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from main import build_run_specs
from scripts.reproducibility.real_scale_swap.aggregate_crossover import (
    BASE_MODES,
    LR_ADJUSTED_MODE,
    build_summary,
    early_log_auc,
    load_assignment,
    plot_curves,
    plot_gradient_ratios,
    plot_mode_by_dataset,
    plot_paired_effect,
    validate_assignments,
)
from scripts.reproducibility.real_scale_swap.aggregate_permutation_campaign import (
    holm_adjust,
    linear_auc,
    plot_by_dataset,
)
from scripts.reproducibility.real_scale_swap.permutation_schedule import (
    HIGH_GROUPS,
    assignment_pair,
    validate_schedule,
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


def test_lr_adjusted_setup_changes_only_learning_rate():
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name="scale_swap",
            overrides=["setups=[original,original_lr_adjusted]"],
        )
    specs = {label: (run_cfg, mode) for label, run_cfg, mode in build_run_specs(cfg, 3)}
    original, original_mode = specs["original"]
    adjusted, adjusted_mode = specs["original_lr_adjusted"]

    assert original_mode == adjusted_mode == "original"
    assert original.train.lr == 1.0e-4
    assert adjusted.train.lr == pytest.approx(1.0e-4 / 50.5)
    original.train.lr = adjusted.train.lr
    assert OmegaConf.to_container(original, resolve=True) == OmegaConf.to_container(
        adjusted, resolve=True
    )


def test_permutation_schedule_is_balanced_and_starts_with_completed_pair():
    validate_schedule()
    assert len(HIGH_GROUPS) == 15

    assignment_a, assignment_b = assignment_pair(1)
    np.testing.assert_array_equal(assignment_a, [1.0] * 4 + [10.0] * 4)
    np.testing.assert_array_equal(assignment_b, [10.0] * 4 + [1.0] * 4)

    assignments = np.stack(
        [
            assignment
            for pair_index in range(1, 16)
            for assignment in assignment_pair(pair_index)
        ]
    )
    np.testing.assert_array_equal((assignments == 10.0).sum(axis=0), [15] * 8)


def test_permutation_statistics_use_linear_auc_and_one_sided_pairs():
    steps = np.array([0, 1, 2])
    curves = np.array([[[1.0, 2.0, 3.0]], [[2.0, 2.0, 2.0]]])
    np.testing.assert_allclose(linear_auc(curves, steps), [[2.0], [2.0]])

    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_permutation_plot_accepts_normalized_control(tmp_path):
    names = [f"dataset_{index}" for index in range(8)]
    steps = np.array([0, 1, 2])
    low = np.ones((15, 8, len(steps)))
    high = np.ones_like(low)
    campaign = {
        "names": names,
        "steps": steps,
        "low": low,
        "high": high,
    }

    plot_by_dataset(
        {"normalized": campaign, "original_lr_adjusted": campaign},
        tmp_path / "comparison.png",
    )

    assert (tmp_path / "comparison.png").exists()


def test_early_log_auc_is_lower_for_faster_curve():
    steps = np.array([0, 1000, 2000, 3000])
    slow = np.array([[[1.0, 0.8, 0.6, 0.5]], [[1.0, 0.9, 0.7, 0.6]]])
    fast = np.array([[[1.0, 0.3, 0.1, 0.05]], [[1.0, 0.4, 0.2, 0.1]]])

    assert np.all(early_log_auc(fast, steps) < early_log_auc(slow, steps))


def test_trainer_rejects_non_finite_metrics():
    with pytest.raises(FloatingPointError, match="validation nMSE at step 10"):
        Trainer._require_finite(np.array([1.0, np.nan]), "validation nMSE", 10)


def test_scale_swap_aggregation_writes_comparison_figures(tmp_path):
    names = np.array([f"dataset_{i}" for i in range(8)])
    steps = np.array([0, 1000, 2000])
    assignment_a = np.array([1.0] * 4 + [10.0] * 4)
    assignment_b = np.array([10.0] * 4 + [1.0] * 4)
    paths = []
    for label, assignment in (("a", assignment_a), ("b", assignment_b)):
        path = tmp_path / label
        path.mkdir()
        paths.append(path)
        for mode in ("normalized", "original", LR_ADJUSTED_MODE):
            nmse = np.empty((2, len(steps), len(names)))
            grad_mag = np.empty_like(nmse)
            for category, scale in enumerate(assignment):
                if mode != "normalized" and scale == 10.0:
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

    modes = BASE_MODES + [LR_ADJUSTED_MODE]
    a = load_assignment(paths[0], modes)
    b = load_assignment(paths[1], modes)
    validate_assignments(a, b, modes)
    plot_curves(a, b, modes, tmp_path / "curves_full.png", end_step=None)
    plot_curves(a, b, modes, tmp_path / "curves_early.png", end_step=2000)
    plot_mode_by_dataset(a, b, "original", tmp_path / "datasets.png")
    plot_paired_effect(a, b, modes, tmp_path / "effect.png")
    plot_gradient_ratios(a, b, modes, tmp_path / "gradients.png")
    summary = build_summary(a, b, modes)

    assert (tmp_path / "curves_full.png").exists()
    assert (tmp_path / "curves_early.png").exists()
    assert (tmp_path / "datasets.png").exists()
    assert (tmp_path / "effect.png").exists()
    assert (tmp_path / "gradients.png").exists()
    assert (
        summary["modes"]["original"]["paired_auc_low_minus_high"]["mean_ci95"]["mean"]
        > 0.0
    )
    assert (
        summary["modes"][LR_ADJUSTED_MODE]["paired_full_auc_low_minus_high"][
            "mean_ci95"
        ]["mean"]
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
        load_assignment(tmp_path, ["normalized"])

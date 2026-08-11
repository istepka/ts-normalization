import numpy as np
import pandas as pd

from src.tsfm_pretraining import timesfm_robust_report as report


def test_scale_effect_and_difference_in_differences_use_seed_replication():
    rows = []
    for seed in (0, 1, 2):
        for dataset in ("a", "b"):
            for condition in ("original", "normalized"):
                effect = 0.4 + 0.1 * seed if condition == "original" else 0.0
                for scale, value in ((1.0, effect), (10.0, 0.0)):
                    rows.append(
                        {
                            "seed": seed,
                            "normalization_mode": "whole_context",
                            "objective": "mse",
                            "condition": condition,
                            "assignment": "A" if scale == 1.0 else "B",
                            "dataset": dataset,
                            "scale": scale,
                            "final_mase": value,
                            "mase_auc": value / 2,
                        }
                    )

    effects = report.pair_scale_effects(pd.DataFrame(rows))
    did = report.difference_in_differences(effects)
    assert np.allclose(
        did["final_mase_did"],
        np.repeat([0.4, 0.5, 0.6], 2),
    )

    summary = report.seed_summary(
        did,
        "final_mase_did",
        ["normalization_mode", "objective"],
    )
    assert summary[0]["n_seeds"] == 3
    assert np.isclose(summary[0]["mean"], 0.5)
    assert summary[0]["ci95_half_width"] > 0

    condition = report.condition_effects(pd.DataFrame(rows))
    condition_summary = report.seed_summary(
        condition,
        "final_mase_normalized_minus_original",
        ["normalization_mode", "objective"],
    )
    assert np.isclose(condition_summary[0]["mean"], -0.25)


def test_linear_auc_uses_elapsed_step_weighting():
    steps = np.array([100, 200, 400])
    values = np.array([0.0, 1.0, 1.0])
    assert np.isclose(report.linear_auc(steps, values), 5 / 6)


def test_find_run_dirs_ignores_analysis_output(tmp_path):
    run_dir = tmp_path / "job_seed0"
    run_dir.mkdir()
    (tmp_path / "job_analysis").mkdir()

    assert report.find_run_dirs(tmp_path, "job") == [run_dir]


def test_inequality_effects_compare_conditions_within_seed_and_assignment():
    rows = []
    for seed in (0, 1):
        for assignment in ("A", "B"):
            for condition, values in (
                ("original", (1.0, 3.0)),
                ("normalized", (2.0, 2.0)),
            ):
                for dataset, mase in zip(("a", "b"), values, strict=True):
                    rows.append(
                        {
                            "seed": seed,
                            "normalization_mode": "whole_context",
                            "objective": "mse",
                            "condition": condition,
                            "assignment": assignment,
                            "dataset": dataset,
                            "final_mase": mase,
                        }
                    )

    inequality = report.per_run_inequality(pd.DataFrame(rows))
    effects = report.inequality_condition_effects(inequality)

    assert np.allclose(effects["mase_gini_normalized_minus_original"], -0.25)
    assert np.allclose(
        effects["mase_iqr_normalized_minus_original"],
        -1.0,
    )


def test_natural_rows_support_condition_and_inequality_effects():
    rows = []
    for seed in (0, 1, 2):
        for condition, values in (
            ("original", (2.0, 4.0)),
            ("normalized", (1.0, 1.0)),
        ):
            for dataset, mase in zip(("a", "b"), values, strict=True):
                rows.append(
                    {
                        "seed": seed,
                        "normalization_mode": "whole_context",
                        "objective": "mse",
                        "condition": condition,
                        "assignment": "natural",
                        "dataset": dataset,
                        "scale": 1.0,
                        "final_mase": mase,
                        "mase_auc": mase,
                    }
                )

    rows = pd.DataFrame(rows)
    condition = report.condition_effects(rows)
    inequality = report.per_run_inequality(rows)
    inequality_effect = report.inequality_condition_effects(inequality)

    assert np.allclose(
        condition["final_mase_normalized_minus_original"],
        [-1.0, -3.0] * 3,
    )
    assert np.allclose(
        inequality_effect["mase_gini_normalized_minus_original"],
        -1 / 6,
    )

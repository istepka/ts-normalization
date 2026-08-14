"""Fast end-to-end checks of train.py's run_moment/run_timesfm loops against
the synthetic tiny_corpus fixture (wandb disabled). The real corpus path is
exercised separately by the Phase 2/5 smoke tests (see notes/agentic_logs)."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import wandb
from src.training import tsfm as train_mod


def _base_cfg(tiny_corpus, tmp_path, model: str) -> OmegaConf:
    root, _ = tiny_corpus
    cfg = OmegaConf.create(
        {
            "model": model,
            "device": "cpu",
            "condition": {
                "moment": "moment_original",
                "timesfm": "timesfm_native_original",
                "chronos2": "chronos2_normalized",
                "moirai2": "moirai2_normalized",
            }[model],
            "experiment_kind": "natural_mixture",
            "scale_assignment": None,
            "scale_b_low": 1.0,
            "scale_b_high": 10.0,
            "seed": 0,
            "output_dir": str(tmp_path / "out"),
            "corpus": {"root": str(root), "datasets": ["synth_a", "synth_b"]},
            "window_index": {
                "context_length": 64,
                "prediction_length": 32,
                "stride": 80,
                "val_series_fraction": 0.25,
                "min_valid_fraction": 0.9,
                "base_seed": 0,
                "max_windows_per_series": None,
                "cache_path": None,
            },
            "dataset_weights": None,
            "moment": {
                "context_length": 64,
                "patch_len": 8,
                "d_model": 16,
                "t5_num_layers": 2,
                "t5_num_heads": 4,
                "t5_d_ff": 32,
                "t5_d_kv": 4,
                "mask_ratio": 0.3,
                "dropout": 0.1,
                "grad_clip_norm": 1.0,
            },
            "timesfm": {
                "config_size": "17m",
                "grad_clip_norm": 1.0,
                "normalization_mode": "first_patch",
                "objective": "combined",
            },
            "chronos2": {
                "context_length": 64,
                "prediction_length": 32,
                "patch_size": 16,
                "d_model": 32,
                "d_kv": 8,
                "d_ff": 64,
                "num_layers": 2,
                "num_heads": 4,
                "dropout_rate": 0.0,
                "initializer_factor": 0.05,
                "quantiles": [0.1, 0.5, 0.9],
                "use_arcsinh": True,
                "grad_clip_norm": 1.0,
            },
            "moirai2": {
                "context_length": 64,
                "predict_horizon": 32,
                "patch_size": 16,
                "d_model": 64,
                "d_ff": 64,
                "num_layers": 2,
                "max_seq_len": 64,
                "attn_dropout_p": 0.0,
                "dropout_p": 0.0,
                "scaling": True,
                "quantile_levels": [0.1, 0.5, 0.9],
                "grad_clip_norm": 1.0,
            },
            "train": {
                "steps": 4,
                "schedule_steps": None,
                "batch_size": 4,
                "lr": 1e-4,
                "optimizer": "sgd",
                "deterministic": True,
                "schedule_seed": 0,
                "eval_every": 2,
                "eval_batches": 1,
                "eval_windows_per_dataset": 2,
                "checkpoint_every": 4,
                "resume_from": None,
            },
            "wandb": {
                "entity": "x",
                "project": "x",
                "mode": "disabled",
                "experiment": "smoke",
            },
        }
    )
    if model == "timesfm":
        cfg.timesfm.config_size = "17m"
        cfg.window_index.context_length = 64
        cfg.window_index.prediction_length = 128
        cfg.moment.context_length = 64
    return cfg


def test_run_moment_end_to_end(tiny_corpus, tmp_path):
    cfg = _base_cfg(tiny_corpus, tmp_path, "moment")
    index = train_mod.resolve_window_index(cfg, tiny_corpus[1])
    run = wandb.init(mode="disabled")
    summary = train_mod.run_moment(cfg, index, run)
    run.finish()

    assert summary["moment_revision"] is not None
    assert set(summary["windows_processed"]["dataset"]) <= {"synth_a", "synth_b"}
    assert (Path(cfg.output_dir) / "checkpoint_step4.pt").is_file()
    assert (Path(cfg.output_dir) / "summary.json").is_file()


def test_run_moment_resumes_after_checkpoint_and_preserves_history(
    tiny_corpus, tmp_path
):
    cfg = _base_cfg(tiny_corpus, tmp_path, "moment")
    cfg.train.steps = 2
    cfg.train.checkpoint_every = 2
    source_dir = tmp_path / "source"
    cfg.output_dir = str(source_dir)
    index = train_mod.resolve_window_index(cfg, tiny_corpus[1])

    run = wandb.init(mode="disabled")
    train_mod.run_moment(cfg, index, run)
    run.finish()

    cfg.train.steps = 4
    cfg.train.checkpoint_every = 4
    cfg.train.resume_from = str(source_dir / "checkpoint_step2.pt")
    cfg.output_dir = str(tmp_path / "continued")
    run = wandb.init(mode="disabled")
    summary = train_mod.run_moment(cfg, index, run)
    run.finish()

    history = json.loads((Path(cfg.output_dir) / "history.json").read_text())
    assert history["step"] == [2, 4]
    assert summary["optimization"] == {
        "steps_attempted": 4,
        "steps_skipped": 0,
        "updates_applied": 4,
    }
    assert (Path(cfg.output_dir) / "checkpoint_step4.pt").is_file()


def test_run_timesfm_end_to_end(tiny_corpus, tmp_path):
    cfg = _base_cfg(tiny_corpus, tmp_path, "timesfm")
    # 17m config has horizon_len=128; give windows a matching prediction_length
    # and a context long enough for at least one 32-length patch beyond it,
    # while still fitting inside the fixture's 200-point synthetic series.
    cfg.window_index.context_length = 64
    cfg.window_index.prediction_length = 128
    cfg.window_index.stride = 200
    cfg.experiment_kind = "controlled_scale"
    cfg.scale_assignment = "A"
    cfg.timesfm.normalization_mode = "whole_context"
    cfg.timesfm.objective = "mse"
    index = train_mod.resolve_window_index(cfg, tiny_corpus[1])
    run = wandb.init(mode="disabled")
    summary = train_mod.run_timesfm(cfg, index, run)
    run.finish()

    assert summary["timesfm_revision"] is not None
    assert set(summary["windows_processed"]["dataset"]) <= {"synth_a", "synth_b"}
    assert (Path(cfg.output_dir) / "checkpoint_step4.pt").is_file()


def test_run_chronos2_end_to_end(tiny_corpus, tmp_path):
    cfg = _base_cfg(tiny_corpus, tmp_path, "chronos2")
    cfg.experiment_kind = "controlled_scale"
    cfg.scale_assignment = "A"
    index = train_mod.resolve_window_index(cfg, tiny_corpus[1])
    run = wandb.init(mode="disabled")
    summary = train_mod.run_chronos2(cfg, index, run)
    run.finish()

    assert summary["chronos2_revision"] == "chronos-forecasting==2.3.1"
    assert set(summary["windows_processed"]["dataset"]) <= {
        "synth_a",
        "synth_b",
    }
    assert (Path(cfg.output_dir) / "checkpoint_step4.pt").is_file()


def test_run_moirai2_end_to_end(tiny_corpus, tmp_path):
    cfg = _base_cfg(tiny_corpus, tmp_path, "moirai2")
    cfg.experiment_kind = "controlled_scale"
    cfg.scale_assignment = "A"
    index = train_mod.resolve_window_index(cfg, tiny_corpus[1])
    run = wandb.init(mode="disabled")
    summary = train_mod.run_moirai2(cfg, index, run)
    run.finish()

    assert summary["moirai2_revision"] is not None
    assert set(summary["windows_processed"]["dataset"]) <= {
        "synth_a",
        "synth_b",
    }
    assert (Path(cfg.output_dir) / "checkpoint_step4.pt").is_file()


def test_history_is_persisted_before_summary_finalization(
    tiny_corpus, tmp_path, monkeypatch
):
    cfg = _base_cfg(tiny_corpus, tmp_path, "moment")
    index = train_mod.resolve_window_index(cfg, tiny_corpus[1])

    def fail_finalization(*args, **kwargs):
        raise RuntimeError("forced finalization failure")

    monkeypatch.setattr(train_mod, "finalize_summary", fail_finalization)
    run = wandb.init(mode="disabled")
    with pytest.raises(RuntimeError, match="forced finalization failure"):
        train_mod.run_moment(cfg, index, run)
    run.finish()

    assert (Path(cfg.output_dir) / "history.json").is_file()


def test_dataset_weights_must_cover_every_present_dataset(tiny_corpus, tmp_path):
    cfg = _base_cfg(tiny_corpus, tmp_path, "moment")
    cfg.dataset_weights = {"synth_a": 1.0}  # missing synth_b
    index = train_mod.resolve_window_index(cfg, tiny_corpus[1])
    try:
        train_mod.resolve_dataset_weights(cfg, index)
    except ValueError as e:
        assert "synth_b" in str(e)
    else:
        raise AssertionError("expected ValueError for incomplete dataset_weights")


def test_finalize_summary_omits_early_auc_for_sparse_eval_schedule():
    history = {
        "step": [2000, 4000],
        "pooled_mse": [2.0, 1.0],
    }
    cfg = OmegaConf.create({"model": "timesfm"})
    optimization = {"steps_attempted": 4000, "steps_skipped": 0}

    summary = train_mod.finalize_summary(history, {}, cfg, optimization)

    assert summary["final_pooled_mse"] == 1.0
    assert "log_mse_auc_through_2000" not in summary


def test_finalize_summary_keeps_early_auc_when_two_points_are_available():
    history = {
        "step": [1000, 2000, 4000],
        "pooled_mse": [4.0, 2.0, 1.0],
    }
    cfg = OmegaConf.create({"model": "timesfm"})
    optimization = {"steps_attempted": 4000, "steps_skipped": 0}

    summary = train_mod.finalize_summary(history, {}, cfg, optimization)

    assert "log_mse_auc_through_2000" in summary

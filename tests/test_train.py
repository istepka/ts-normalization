"""Fast end-to-end checks of train.py's run_moment/run_timesfm loops against
the synthetic tiny_corpus fixture (wandb disabled). The real corpus path is
exercised separately by the Phase 2/5 smoke tests (see notes/agentic_logs)."""

from pathlib import Path

from omegaconf import OmegaConf

import wandb
from src.tsfm_pretraining import train as train_mod


def _base_cfg(tiny_corpus, tmp_path, model: str) -> OmegaConf:
    root, _ = tiny_corpus
    cfg = OmegaConf.create(
        {
            "model": model,
            "device": "cpu",
            "condition": "moment_original"
            if model == "moment"
            else "timesfm_native_original",
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
            "timesfm": {"config_size": "17m", "grad_clip_norm": 1.0},
            "train": {
                "steps": 4,
                "batch_size": 4,
                "lr": 1e-4,
                "optimizer": "sgd",
                "schedule_seed": 0,
                "eval_every": 2,
                "eval_batches": 1,
                "checkpoint_every": 4,
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


def test_run_timesfm_end_to_end(tiny_corpus, tmp_path):
    cfg = _base_cfg(tiny_corpus, tmp_path, "timesfm")
    # 17m config has horizon_len=128; give windows a matching prediction_length
    # and a context long enough for at least one 32-length patch beyond it,
    # while still fitting inside the fixture's 200-point synthetic series.
    cfg.window_index.context_length = 64
    cfg.window_index.prediction_length = 128
    cfg.window_index.stride = 200
    index = train_mod.resolve_window_index(cfg, tiny_corpus[1])
    run = wandb.init(mode="disabled")
    summary = train_mod.run_timesfm(cfg, index, run)
    run.finish()

    assert summary["timesfm_revision"] is not None
    assert set(summary["windows_processed"]["dataset"]) <= {"synth_a", "synth_b"}
    assert (Path(cfg.output_dir) / "checkpoint_step4.pt").is_file()


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

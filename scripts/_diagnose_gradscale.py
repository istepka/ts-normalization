"""One-off: quantify the optimization confound between loss spaces.

Both MOMENT arms trained with lr=1e-4 and grad_clip_norm=1.0. Original-space
MSE gradients scale as b^2, so if they sit far above the clip threshold the
original arm is clip-saturated and the two arms are not running comparable
optimization. Measures, on identical batches from a fresh init, the
pre-clip gradient norm under each loss space.
"""

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tsfm_pretraining import losses as L
from src.tsfm_pretraining import moment_adapter as ma
from src.tsfm_pretraining import train as T
from src.tsfm_pretraining import window_index as wi

RUN = "outputs/gifteval_moment_26371_natural_moment_normalized"
cfg = OmegaConf.load(f"{RUN}/resolved_config.yaml")
index = T.resolve_window_index(cfg)
cache = wi.SeriesCache(index.corpus_root)
model_cfg = ma.MomentConfig(**OmegaConf.to_container(cfg.moment, resolve=True))

BATCH = 64  # debug-partition GPUs are ~10GB; the ratio does not need full batches
weights = T.resolve_dataset_weights(cfg, index)
schedule = wi.build_batch_schedule(
    index, "train", weights, 8, BATCH, cfg.train.schedule_seed
)
train_table = index.split("train").reset_index(drop=True)
clip = cfg.moment.grad_clip_norm

stats = {"moment_normalized": [], "moment_original": []}
for positions in schedule:
    rows = train_table.iloc[positions]
    batch = ma.make_batch(index, rows, cache).to("cuda")
    for condition in stats:
        # Fresh init per measurement so the two conditions see identical
        # parameters, not parameters already moved by the other condition.
        model = ma.build_moment_model(model_cfg, seed=cfg.seed).to("cuda")
        model.zero_grad(set_to_none=True)
        result = ma.forward(model, batch, condition)
        result.per_example_loss_masked.mean().backward()
        stats[condition].append(L.grad_norm(model.parameters()))

for condition, values in stats.items():
    v = np.array(values)
    print(
        f"{condition:20s} grad_norm med={np.median(v):.4e} "
        f"min={v.min():.4e} max={v.max():.4e} "
        f"frac_above_clip({clip})={np.mean(v > clip):.2f}"
    )

n = np.median(stats["moment_normalized"])
o = np.median(stats["moment_original"])
print(f"\nmedian original/normalized gradient-norm ratio: {o / n:.3e}")
print(f"implied lr adjustment for original space: lr * {n / o:.3e}")

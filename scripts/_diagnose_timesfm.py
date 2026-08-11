"""One-off: why did timesfm_normalized training destabilize?

Checks, on real TRAIN windows, whether the normalized-space loss is driven by
a near-zero sigma (the clamp hypothesis) or by target outliers relative to the
context scale.
"""

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tsfm_pretraining import timesfm_model as tm
from src.tsfm_pretraining import train as T
from src.tsfm_pretraining import window_index as wi

RUN = "outputs/gifteval_timesfm_26378_natural_timesfm_normalized"
cfg = OmegaConf.load(f"{RUN}/resolved_config.yaml")
index = T.resolve_window_index(cfg)
cache = wi.SeriesCache(index.corpus_root)
model_cfg = T.TIMESFM_CONFIGS[cfg.timesfm.config_size]

# Fresh init: we want the loss landscape the run STARTED from, not a model
# already wrecked by it.
model = tm.build_timesfm_model(model_cfg, seed=cfg.seed).to("cuda")
model.eval()

weights = T.resolve_dataset_weights(cfg, index)
schedule = wi.build_batch_schedule(
    index, "train", weights, 4, 512, cfg.train.schedule_seed
)
train_table = index.split("train").reset_index(drop=True)

sig, nt_max, mse_n, mse_o, ds_all = [], [], [], [], []
for positions in schedule:
    rows = train_table.iloc[positions]
    batch = tm.make_batch(index, rows, cache, model_cfg.horizon_len).to("cuda")
    with torch.no_grad():
        _, _, (mu, sigma) = tm.run_decoder(model, batch)
        out_n = tm.forward(model, batch, "timesfm_normalized")
        out_o = tm.forward(model, batch, "timesfm_native_original")
        norm_target = (batch.target - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
    sig.append(sigma.cpu().numpy())
    nt_max.append(norm_target.abs().max(dim=1).values.cpu().numpy())
    mse_n.append(out_n.mse_per_example.cpu().numpy())
    mse_o.append(out_o.mse_per_example.cpu().numpy())
    ds_all.append(batch.dataset)

sig = np.concatenate(sig)
nt_max = np.concatenate(nt_max)
mse_n = np.concatenate(mse_n)
ds_all = np.concatenate(ds_all)

print(f"n windows: {sig.size}")
print(
    f"sigma      min={sig.min():.3e} p01={np.percentile(sig, 1):.3e} med={np.median(sig):.3e}"
)
print(f"  at clamp floor (<=1e-6): {(sig <= 1.000001e-6).sum()}")
print(f"  sigma < 1e-3: {(sig < 1e-3).sum()}")
print(
    f"|normalized target| max: med={np.median(nt_max):.3e} p99={np.percentile(nt_max, 99):.3e} max={nt_max.max():.3e}"
)
print(
    f"normalized MSE: med={np.median(mse_n):.3e} p99={np.percentile(mse_n, 99):.3e} max={mse_n.max():.3e}"
)

k = np.argsort(-mse_n)[:10]
print("\ntop-10 normalized-space loss windows:")
for i in k:
    print(
        f"  {str(ds_all[i]):36s} mse_n={mse_n[i]:.3e} sigma={sig[i]:.3e} "
        f"|norm_target|max={nt_max[i]:.3e}"
    )

frac = mse_n[np.argsort(-mse_n)[: max(1, mse_n.size // 100)]].sum() / mse_n.sum()
print(f"\ntop 1% of windows contribute {frac:.1%} of total normalized-space loss")

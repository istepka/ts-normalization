"""One-off: find what inflates per-dataset nMSE on the real corpus."""

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tsfm_pretraining import moment_adapter as ma
from src.tsfm_pretraining import train as T
from src.tsfm_pretraining import window_index as wi

RUN = "outputs/gifteval_moment_26371_scale_moment_original_B"
cfg = OmegaConf.load(f"{RUN}/resolved_config.yaml")
index = T.resolve_window_index(cfg)
cache = wi.SeriesCache(index.corpus_root)

rows = T.sample_stratified_eval_rows(index, cfg.train.eval_windows_per_dataset)
batch = ma.make_batch(index, rows, cache)
model_cfg = ma.MomentConfig(**OmegaConf.to_container(cfg.moment, resolve=True))
model = ma.build_moment_model(model_cfg, seed=cfg.seed).to("cuda")
state = torch.load(f"{RUN}/checkpoint_step30000.pt", map_location="cuda")
model.load_state_dict(state["model"])
model.eval()

nmse_parts, std_parts = [], []
for start in range(0, len(rows), 256):
    chunk = ma.slice_rows = batch
    sl = slice(start, min(start + 256, len(rows)))
    sub = ma.MomentBatch(
        x_enc=batch.x_enc[sl],
        input_mask=batch.input_mask[sl],
        dataset=batch.dataset[sl],
        domain=batch.domain[sl],
        frequency=batch.frequency[sl],
        scale=batch.scale[sl],
        batch_seed=batch.batch_seed,
    ).to("cuda")
    with torch.no_grad():
        out = ma.forward(model, sub, cfg.condition)
    nmse_parts.append(out.normalized_mse.cpu().numpy())
    # per-window context std actually used by RevIN
    std_parts.append(sub.x_enc.squeeze(1).std(dim=1).cpu().numpy())

nmse = np.concatenate(nmse_parts)
stds = np.concatenate(std_parts)
ds = rows["dataset"].to_numpy()

print(f"nMSE  min={nmse.min():.3e} med={np.median(nmse):.3e} max={nmse.max():.3e}")
print(f"ctx std min={stds.min():.3e} med={np.median(stds):.3e}")
print(f"windows with std < 1e-6: {(stds < 1e-6).sum()} / {stds.size}")
print(f"windows with nMSE > 100: {(nmse > 100).sum()}")

per = {d: float(nmse[ds == d].mean()) for d in np.unique(ds)}
print("\ntop 8 datasets by mean nMSE:")
for d, v in sorted(per.items(), key=lambda kv: -kv[1])[:8]:
    sel = ds == d
    print(f"  {d:40s} nmse={v:.4e}  min_std={stds[sel].min():.3e}")

order = np.argsort(-nmse)[:8]
print("\ntop 8 individual windows:")
for i in order:
    print(f"  {ds[i]:40s} nmse={nmse[i]:.4e} ctx_std={stds[i]:.4e}")

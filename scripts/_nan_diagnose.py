import numpy as np
from pathlib import Path

from src.tsfm_pretraining import gifteval_corpus as gc
from src.tsfm_pretraining import moment_adapter as ma
from src.tsfm_pretraining import window_index as wi

root = Path("/zfsauton/scratch/istepka/lts/data/giftevalpretrain_full")
dmap = gc.load_domain_map()

config = wi.WindowIndexConfig(
    context_length=512, prediction_length=128, stride=512, base_seed=0
)
index = wi.WindowIndex.load(
    Path("outputs/gifteval_window_index/context512_pred128.parquet"), config, root
)

dataset_weights = {d: 1.0 for d in index.table["dataset"].unique()}
schedule = wi.build_batch_schedule(
    index, "train", dataset_weights, steps=5, batch_size=512, schedule_seed=0
)
train_table = index.split("train").reset_index(drop=True)
cache = wi.SeriesCache(root)

for step in range(5):
    rows = train_table.iloc[schedule[step]]
    batch = ma.make_batch(
        index, rows, cache, scale_assignment="B", b_low=1.0, b_high=10.0
    )
    x = batch.x_enc.numpy()
    print(
        f"step {step}: x_enc min={x.min():.3e} max={x.max():.3e} "
        f"abs_max={np.abs(x).max():.3e} any_inf={np.isinf(x).any()} any_nan={np.isnan(x).any()}"
    )
    worst_idx = np.argmax(np.abs(x).max(axis=(1, 2)))
    worst_row = rows.iloc[worst_idx]
    print(
        f"  worst window: dataset={worst_row['dataset']} series={worst_row['series_id']} "
        f"scale={float(batch.scale[worst_idx]):.1f} context_mean={worst_row['context_mean']:.3e} "
        f"context_std={worst_row['context_std']:.3e}"
    )

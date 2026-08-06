# GiftEvalPretrain MOMENT/TimesFM loss-space implementation

Implements notes/05-timesfm-pretraining-loss-space-plan.md: MOMENT masked-reconstruction
and reduced TimesFM-1.0 forecasting pretraining on GiftEvalPretrain, with normalized-space
vs original-space loss conditions, dispersion/equity metrics, and wandb tracking. New
module `src/tsfm_pretraining/` plus `scripts/{audit,build_gifteval_window_index,
run_moment_pretraining,run_timesfm_pretraining,aggregate_tsfm_loss_space}*`. Does not touch
the existing PatchTST loss-space code (`src/`, `main.py`, `conf/config.yaml`).

## Corpus location

The plan assumed GiftEvalPretrain was already local; it was not (only an empty HF cache
stub existed). The full 836GB, 152-directory corpus turned out to already exist at
`/zfsauton/scratch/istepka/lts/data/giftevalpretrain_full/` (found from a user pointer, not
autodiscovered). `conf/tsfm_moment.yaml` / `conf/tsfm_timesfm.yaml` point `corpus.root`
there. GiftEvalPretrain's "71 univariate and 17 multivariate" dataset-card count is a
logical-family count; the physical corpus has 152 directories because `era5_YYYY` (30) and
`cmip6_YYYY` (33) are split per year. `gifteval_corpus.py` treats each physical directory as
one "dataset" source, which is a coarser-than-family but simpler and storage-faithful unit.

## Domain mapping

The corpus carries no per-dataset domain field, and no authoritative public per-directory
domain table exists for GiftEvalPretrain specifically (only GIFT-Eval's 28-benchmark-config
`notebooks/dataset_properties.json`, which doesn't cover the pretrain corpus's directories).
`src/tsfm_pretraining/gifteval_domains.yaml` is a manually curated dataset->domain mapping
using GIFT-Eval's confirmed 7-domain taxonomy (Econ/Fin, Energy, Healthcare, Nature, Sales,
Transport, Web/CloudOps), verified to exactly cover all 152 real directories
(`test_domain_map_covers_real_corpus_exactly` in `tests/test_gifteval_corpus.py`, plus a
direct diff against the real corpus during implementation). ~126/152 entries are high
confidence (well-known benchmark families); 18 are medium and 8 are low confidence
(`borealis`, `bull`, `cockatoo`, `hog`, `pdb`, `sceaux`, `smart`, `spain` -- names too
generic to place confidently from the name alone). `gifteval_corpus.describe_dataset` fails
loudly (`KeyError`) on any directory missing from this map rather than guessing; the audit
script surfaces low-confidence entries in its summary so a human can correct them.

## MOMENT and TimesFM: vendored, not pip-installed

Reused the official model code per the plan's "reuse ... add a project adapter" instruction,
but vendored the source (`src/tsfm_pretraining/vendor/{moment,timesfm_v1}/`, pinned commit
SHAs and license notes in each `REVISION` file) instead of `pip install`ing, because
momentfm's published pins (`numpy==1.25.2`, `transformers==4.33.3`,
`huggingface-hub==0.24.0`) hard-conflict with this project's dependencies. The vendored
MOMENT code needed two intentional deviations from upstream (documented in
`vendor/moment/REVISION`): dropped unused task heads/embeddings, and `reconstruction()` now
also returns the pre-inverse-transform normalized decoder output and RevIN stats via
`TimeseriesOutputs.metadata`, since `moment_adapter.py` needs those to compute the
`moment_normalized` condition without duplicating MOMENT's forward pass. Both vendored
models import cleanly against the current dependency stack (transformers 5.x,
huggingface-hub 1.x) with zero compatibility shims needed -- confirmed by direct
forward/backward smoke tests before writing any adapter code.

For TimesFM, `google-research/timesfm`'s legacy `v1/src/timesfm/pytorch_patched_decoder.py`
is Google's own official PyTorch port of the exact architecture the plan lists (causal
decoder, patch 32/128, instance norm, residual blocks, quantile+point heads) -- vendored
byte-for-byte, unmodified. Neither model uses its publicly released pretrained checkpoint;
both train from a random initialization on GiftEvalPretrain, per the plan's "will not claim
to reproduce published ... checkpoints" framing. `timesfm_model.py`'s reduced configs:
`CONFIG_17M` (10 layers, hidden 512, head_dim 64, ~17.7M params) and `CONFIG_70M` (12
layers, hidden 960, head_dim 80 matching the official 200M checkpoint's head_dim, ~70.9M
params), found by a small brute-force parameter-count search against
`PatchedTimeSeriesDecoder`'s actual instantiated size.

## Two real bugs found and fixed during testing

1. **`losses.masked_mse` silent shape bug.** MOMENT's context tensor is `[B, 1, L]`
   (channel dim = 1) but the training mask was `[B, L]`; multiplying them broadcasts into a
   `[B, B, L]` cross product instead of per-example masking (PyTorch broadcasting pads the
   shorter shape on the *left*, so the mask's batch dim lands against the pred's channel
   dim). Found by a test that recomputed `moment_normalized`'s loss by hand from the model's
   own RevIN stats and got a different number than the adapter. Fixed at the call site
   (`moment_adapter.forward` unsqueezes the mask) and added a `mask.ndim != pred.ndim`
   guard in `masked_mse` itself so this class of bug fails loudly instead of silently
   computing a wrong-shaped result.
2. **Vendored `TimesFMAttention.scaling` is uninitialized memory.** Google's code allocates
   it with `torch.empty(...)` and never initializes it -- harmless when loading their
   released checkpoint (every parameter gets overwritten), but genuine uninitialized memory
   when training from scratch, which this project always does. Demonstrated directly:
   dirtying the allocator with a large tensor before construction produced scaling values up
   to `4.7e+30`, which explode through `softplus` into the attention scale and intermittently
   produced NaN losses (surfaced as a one-off flaky test failure that would not reproduce
   until deliberately provoked). Fixed in `timesfm_model.build_timesfm_model` (not the
   vendored file) by zero-initializing `scaling` after construction, with a regression test
   that dirties the allocator the same way before building a model.

## Verification performed

- 52 pytest tests (`tests/test_{gifteval_corpus,window_index,losses,moment_adapter,
  timesfm_model,train}.py`) against a synthetic GiftEvalPretrain-shaped fixture
  (`tests/conftest.py`), covering: domain map completeness, univariate/multivariate
  detection (including a real cross-version `datasets` library schema difference -- the
  installed `datasets==5.0.1` names a nested-sequence feature `"List"`, the real corpus
  (written by an older `datasets` version) names it `"Sequence"`; `is_univariate` now
  accepts both), series-level train/val split disjointness, exact scale-assignment
  complementarity, seeded batch/mask replay determinism, masked-reconstruction shapes, the
  MOMENT `b**2` and TimesFM `b**2`(MSE)/`b`(pinball) original-space controlled-scale gradient
  ratios (all within 2% of the theoretical value) plus the complementary acceptance
  criterion that normalized-space removes that dependence (both models' normalized-space
  gradient ratio for the same b=1/b=10 pair is within 2% of 1.0, not 100), TimesFM
  causal-attention/inverse-transform-affine correctness, and full `train.py`
  `run_moment`/`run_timesfm` loops end to end.
- Real-corpus smoke tests (Phase 1/2/5) against `fred_md`, `PEMS03`, `nn5_weekly`: audit
  script, MOMENT and TimesFM-17M `train.py` runs at production hyperparameters (15-200
  steps), checkpoint save/load for both models, and full determinism replay at the
  `train.py` entrypoint level (two independent runs of the same config produce
  byte-identical `summary.json`, aside from wall-clock time).
- GPU verification on `rhea`'s Slurm cluster (`srun --partition=debug --qos=qos_debug
  --gres=gpu:1`, RTX 2080 Ti): the full 50-test pytest suite, and both `train.py` entrypoints
  with `device=cuda`, confirmed correct. This also caught that batch tensors and the
  `torch.random.fork_rng` call in `moment_adapter.forward` (MOMENT's masking RNG draws on
  `x_enc.device`) needed to be device-aware; fixed before this was exercised on GPU, not
  after -- `MomentBatch`/`TimesFMBatch` gained a `.to(device)` method and `train.py` calls it
  on every batch.

## Known limitations / scope decisions

- **Short low-frequency series lose all windows.** `window_length = context_length +
  prediction_length` (512+128=640 by default, chosen so context is exactly 16 TimesFM
  patches and prediction is exactly one 128-point output patch). Weekly/monthly/quarterly
  datasets are often far shorter (e.g. `nn5_weekly` series are only 105 points) and
  contribute zero windows at this config -- an inherent consequence of preserving TimesFM's
  fixed patch/horizon structure, not a bug. `build_gifteval_window_index.py` and `train.py`
  both print an explicit warning listing which requested datasets ended up contributing zero
  windows, and `train.py` records `windows_per_dataset` / `zero_window_datasets` in
  `window_index_meta.json`, so this is visible rather than silently discovered later. A
  smaller `context_length` would recover more of the corpus's frequency diversity at the
  cost of matching TimesFM's official patch structure less closely; this is a real tradeoff
  to revisit before the full run, not resolved here.
- **Full-corpus window-index build time is not fully characterized.** Row-by-row Arrow
  access (`ds[i]` per series) did not finish scanning `buildings_900k` alone (1.8M series) in
  90s; switched `window_index.build_window_index` to the same batched `ds[start:start+
  BATCH_SIZE]` slicing `gifteval_corpus.iter_series_records` already used, which completed
  the same `buildings_900k` scan in a few minutes (still the single largest dataset in the
  corpus by series count). The full 152-directory corpus was not scanned start to finish in
  this session (see the sizing note in `scripts/run_{moment,timesfm}_pretraining.sbatch` and
  the dedicated `scripts/build_gifteval_window_index.sbatch` prerequisite job, which exists
  specifically so this CPU/IO-bound cost isn't paid on an expensive GPU allocation). Also
  fixed a related waste: `build_window_index` was computing each dataset's full-file SHA256
  checksum (`describe_dataset`'s default) even though the window index never stores or uses
  it -- `describe_dataset(..., compute_checksum=False)` skips this; checksumming stays the
  audit script's job, where the provenance record is actually used.
- **`uvx ruff format`/`--fix` mechanically touched the vendored files** (`Optional[X]` ->
  `X | None`, `super(Class, self)` -> `super()`, import sorting, etc.), which would have
  quietly broken the "byte-for-byte" / "no functional changes" claims in both `REVISION`
  files. Restored both vendor subtrees from freshly re-fetched upstream source (verified
  identical to the exact two documented deviations in `vendor/moment/REVISION` for
  `models/moment.py`, byte-identical elsewhere) and added `[tool.ruff] extend-exclude =
  ["src/tsfm_pretraining/vendor"]` to `pyproject.toml` so this can't recur. Confirmed the
  exclude only applies to implicit directory walks (`ruff check src/`), not paths passed
  explicitly on the CLI -- expected ruff behavior, not a gap.
- **GPU throughput was measured, not the full 30k-step runs.** 200 steps on an RTX 2080 Ti:
  MOMENT default config ~9.4s, TimesFM-70M ~20.5s (both single-GPU, real corpus data),
  extrapolating to roughly 24 and 51 minutes per 30k-step run respectively -- fast enough
  that throughput is not a concern for the "70M primary run if throughput permits" plan
  instruction, but not verified end to end at 30k steps in this session.
- Two loss-space conditions run per model per experiment kind (natural-mixture,
  controlled-scale). Originally `scripts/run_{moment,timesfm}_pretraining.sbatch` ran all 6
  combinations (2 natural-mixture + 2 conditions x 2 scale assignments) sequentially in one
  Slurm job; per user request this was restructured into a 6-task Slurm job array (one GPU,
  one run per task, run in parallel subject to GPU availability) plus a separate
  `scripts/aggregate_{moment,timesfm}_pretraining.sbatch` submitted with
  `--dependency=afterok:<array_job_id>`, wired together by
  `scripts/submit_{moment,timesfm}_pretraining.sh`. This removes the original design's
  biggest operational risk (one task failing partway through no longer takes the other 5
  down with it) and turns 6x sequential wall-clock into ~1x. A single failed array task can
  be resubmitted against the same output namespace via `JOBTAG=<original tag> sbatch
  --array=<index> scripts/run_..._pretraining.sbatch` (documented in each submit script).
  **Gotcha hit while testing this**: `sbatch --export=ALL,VAR=value,...` (needed, it seemed,
  to forward variables `export`ed in the wrapper script) put the array job into a
  `user_env_retrieval_failed_requeued_held` state requiring a manual `scontrol release`.
  Fixed by dropping `--export` entirely -- `sbatch` already forwards the submitting shell's
  full environment by default, so the flag was both redundant and the actual cause. Verified
  clean (no held state) on a real submission after the fix.
- **MOMENT's default backbone size was an arbitrary shrink, not calibrated to any real
  MOMENT checkpoint** (a bug, not a deliberate choice): `d_model=256, 4 T5 layers` picked
  without checking against official sizes, ~2.1M params. Also found and fixed a real
  validation bug while investigating this -- `MomentConfig.__post_init__` rejected any
  config where `d_model % t5_num_heads != 0`, which is not a real T5 constraint (T5's
  attention head dimension is `d_kv`, independent of `d_model/num_heads`) and would have
  incorrectly rejected the actual `google/flan-t5-small` config (`d_model=512,
  num_heads=6`). After fixing that, `conf/tsfm_moment.yaml` now defaults to MOMENT-base
  dims (`d_model=768, 12 layers, 12 heads, d_ff=2048, d_kv=64`, matching
  `T5Config.from_pretrained("google/flan-t5-base")` exactly) -> ~66.1M params, per user
  choice between small (~14.7M) / base (~66.1M, chosen) / large (~239.2M). TimesFM stays at
  the plan's specified 70M per user confirmation (the plan deliberately reduces TimesFM's
  size for tractability; that wasn't an oversight worth revisiting).
- **Batch size: "maximize throughput" and "maximize VRAM utilization" are not the same
  target here, and pointed in opposite directions.** Probed real batch-size/VRAM/throughput
  curves on this cluster's actual H200 (150GB) with MOMENT-base and TimesFM-70M against real
  corpus data. Windows/sec plateaus early and stays flat long before 85% VRAM (e.g. MOMENT-
  base: batch 256 -> 1415 windows/s at 8.5% VRAM, batch 3072 -> 1581 windows/s, +12%, at 95%
  VRAM). More importantly, `train.steps` is the fixed unit across paired conditions (not
  epochs or wall-clock time), and steps/sec strictly *decreases* as batch size grows (each
  step does more work) -- so a bigger batch makes a fixed-step-count run slower in wall
  clock, not faster, directly opposing a naive "fill the GPU" instinct. Settled on
  `batch_size=512` for both models (per user decision after seeing this tradeoff laid out):
  captures most of the achievable per-step throughput, keeps VRAM usage low enough to be a
  safe neighbor on a shared 8-GPU node, and keeps a single 30k-step run's wall clock in the
  range of roughly an hour to a few hours rather than the 14-18 hours a literal 85%-VRAM
  batch size would cost for a few percent of extra data throughput. `BATCH_SIZE` is
  overridable per submission if this should be re-tuned later (e.g. after a model size
  change).
- `scripts/aggregate_tsfm_loss_space.py` produces comparison tables and the paired
  per-dataset AUC effect (JSON/CSV) but no figures, following this project's existing
  separation between aggregate scripts (numbers) and replot scripts (figures); figure
  generation was not part of this pass.

## Files added

`src/tsfm_pretraining/{__init__,gifteval_corpus,gifteval_domains.yaml,window_index,losses,
moment_adapter,timesfm_model,train}.py`, `src/tsfm_pretraining/vendor/{moment,timesfm_v1}/`,
`scripts/{audit_gifteval_pretrain,build_gifteval_window_index,aggregate_tsfm_loss_space}.py`,
`scripts/build_gifteval_window_index.sbatch`, `scripts/run_{moment,timesfm}_pretraining.sbatch`
(job arrays), `scripts/aggregate_{moment,timesfm}_pretraining.sbatch`,
`scripts/submit_{moment,timesfm}_pretraining.sh` (the wrapper scripts to actually submit --
they chain index build -> array job -> aggregation with the right Slurm dependencies),
`conf/tsfm_{moment,timesfm}.yaml`, `tests/{conftest,test_gifteval_corpus,test_window_index,
test_losses,test_moment_adapter,test_timesfm_model,test_train}.py`. Added `datasets`,
`pyarrow`, `transformers` (now direct, was transitive) to `pyproject.toml`, and `pytest` as a
dev dependency (the existing test suite had no way to actually run before this: bare `uv run
pytest` resolves to a conda `pytest` outside the project venv on this machine, and pytest
wasn't installed in the venv at all -- `uv run python -m pytest` is the working invocation,
now used throughout).

#!/usr/bin/env bash
#SBATCH --job-name=gifteval_pretraining
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out

# Shared GiftEvalPretrain loss-space launcher for MOMENT, TimesFM, Chronos-2,
# and Moirai-2.0 (MODEL=moment|timesfm|chronos2|moirai2). Each task runs one
# (condition, experiment_kind[, scale_assignment]) run, so all tasks execute
# in parallel (subject to GPU availability) instead of sequentially, and a
# single task failing doesn't take the others down with it. MOMENT and
# TimesFM run a single SEED per array job (6-task array, the #SBATCH
# --array default above); Chronos-2 and Moirai-2.0 sweep SEEDS_CSV in the
# same array (6 runs per seed, 24 tasks for the default 4 seeds -- pass
# --array=0-23%8 when submitting directly). Prefer
# scripts/submit_pretraining.sh over submitting this directly -- it sets
# MODEL, the array bounds, and (for MOMENT/TimesFM) the prerequisite index
# build and dependent aggregation job for you. For MODEL=timesfm, each task
# independently runs the architecture parity/smoke test suite first -- "Do
# not begin TimesFM pretraining until the parity test passes" is an explicit
# plan instruction, so a task refuses to proceed if it fails. This script
# must never be executed on the Rhea login node.
#
# Requires a pre-built window index (INDEX below) -- this does NOT build it
# inline as a fallback: parallel tasks racing to build and write the same
# cache file would be a correctness risk, not just wasted work. Build it
# once first:
#   sbatch scripts/build_gifteval_window_index.sbatch
# (scripts/submit_pretraining.sh does this automatically if missing.)
#
# Tuned against the real corpus on this cluster's H200 (see
# notes/agentic_logs): every model plateaus in windows/sec well before 85%
# VRAM utilization -- pushing batch size further trades away gradient-update
# frequency for a shrinking throughput return and risks OOM. Every model
# defaults to 512, which all four have trained at; override with
# BATCH_SIZE=<n>. TimesFM's
# window_index prediction_length must stay 128 to match every TimesFM
# config's horizon_len; only context_length is meant to be tuned.

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs
source scripts/output_paths.sh

case "${MODEL:-}" in
  moment | timesfm | chronos2 | moirai2) ;;
  *)
    echo "ERROR: MODEL must be set to moment, timesfm, chronos2, or moirai2" >&2
    exit 1
    ;;
esac

if [ "${MODEL}" = "chronos2" ] || [ "${MODEL}" = "moirai2" ]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi

if [ "${MODEL}" = "timesfm" ]; then
  echo "running TimesFM architecture parity/smoke tests before pretraining ..."
  uv run python -m pytest tests/test_timesfm_model.py -q
fi

CONTEXT_LENGTH=${CONTEXT_LENGTH:-512}
PREDICTION_LENGTH=${PREDICTION_LENGTH:-128}
RUN_DATE=${RUN_DATE:-$(date +%F)}
# The window index is a shared cache, not a run artifact: every paired run must
# read the same file, so it lives at a stable path rather than a dated one.
INDEX_DIR=${INDEX_DIR:-outputs/gifteval_window_index}
INDEX=${INDEX:-${INDEX_DIR}/context${CONTEXT_LENGTH}_pred${PREDICTION_LENGTH}_heldout.parquet}
STEPS=${STEPS:-30000}
BATCH_SIZE=${BATCH_SIZE:-512}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-6000}
EVAL_EVERY=${EVAL_EVERY:-250}
EVAL_BATCHES=${EVAL_BATCHES:-50}
EVAL_WINDOWS_PER_DATASET=${EVAL_WINDOWS_PER_DATASET:-64}
SEED=${SEED:-0}
SEEDS_CSV=${SEEDS_CSV:-0,1,2,3}
CONFIG_SIZE=${CONFIG_SIZE:-70m}
JOBTAG=${JOBTAG:-gifteval_${MODEL}_${SLURM_ARRAY_JOB_ID}}
RUN_ROOT="$(output_path experiments "tsfm_pretraining/${MODEL}" "${JOBTAG}")"

if [ ! -f "${INDEX}" ]; then
  echo "ERROR: window index not found at ${INDEX}." >&2
  echo "Build it first: sbatch scripts/build_gifteval_window_index.sbatch" >&2
  echo "(or submit through scripts/submit_pretraining.sh, which does this for you)" >&2
  exit 1
fi

# task -> (condition, experiment_kind, scale_assignment) mapping. The two
# conditions are always this model's normalized and original-space arms.
case "${MODEL}" in
  moment)
    RUNS=(
      "moment_normalized natural_mixture natural"
      "moment_original natural_mixture natural"
      "moment_normalized controlled_scale A"
      "moment_normalized controlled_scale B"
      "moment_original controlled_scale A"
      "moment_original controlled_scale B"
    )
    ;;
  timesfm)
    RUNS=(
      "timesfm_native_original natural_mixture natural"
      "timesfm_normalized natural_mixture natural"
      "timesfm_native_original controlled_scale A"
      "timesfm_native_original controlled_scale B"
      "timesfm_normalized controlled_scale A"
      "timesfm_normalized controlled_scale B"
    )
    ;;
  chronos2 | moirai2)
    RUNS=(
      "${MODEL}_normalized natural_mixture natural"
      "${MODEL}_original natural_mixture natural"
      "${MODEL}_normalized controlled_scale A"
      "${MODEL}_normalized controlled_scale B"
      "${MODEL}_original controlled_scale A"
      "${MODEL}_original controlled_scale B"
    )
    ;;
esac

# The two natural_mixture runs lead every RUNS list, so NATURAL_ONLY=1 drops
# the four controlled_scale (scale-swap) runs by truncation. Array bounds must
# shrink to match, which scripts/submit_pretraining.sh handles.
NATURAL_ONLY=${NATURAL_ONLY:-0}
if [ "${NATURAL_ONLY}" = "1" ]; then
  RUNS=("${RUNS[@]:0:2}")
fi

if [ "${MODEL}" = "chronos2" ] || [ "${MODEL}" = "moirai2" ]; then
  IFS=, read -r -a SEEDS <<< "${SEEDS_CSV}"
  seed_index=$((SLURM_ARRAY_TASK_ID / ${#RUNS[@]}))
  run_index=$((SLURM_ARRAY_TASK_ID % ${#RUNS[@]}))
  SEED=${SEEDS[$seed_index]}
else
  run_index=$SLURM_ARRAY_TASK_ID
fi
read -r CONDITION KIND ASSIGNMENT <<< "${RUNS[$run_index]}"

if [ "${KIND}" = "controlled_scale" ]; then
  extra_args=(experiment_kind=controlled_scale "scale_assignment=${ASSIGNMENT}")
else
  extra_args=(experiment_kind=natural_mixture)
fi

if [ "${MODEL}" = "chronos2" ] || [ "${MODEL}" = "moirai2" ]; then
  run_dir="${RUN_ROOT}/seed${SEED}_${CONDITION}_${ASSIGNMENT}"
elif [ "${KIND}" = "controlled_scale" ]; then
  run_dir="${RUN_ROOT}/scale_${CONDITION}_${ASSIGNMENT}"
else
  run_dir="${RUN_ROOT}/natural_${CONDITION}"
fi

if [ "${MODEL}" = "timesfm" ]; then
  extra_args+=("timesfm.config_size=${CONFIG_SIZE}")
else
  extra_args+=("seed=${SEED}" "train.schedule_seed=${SEED}")
fi

echo "model=${MODEL} task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} condition=${CONDITION} kind=${KIND} assignment=${ASSIGNMENT} -> ${run_dir}"

PYTHONPATH=. uv run python -m src.training.tsfm \
  --config-name="tsfm_${MODEL}" \
  device=cuda \
  condition="${CONDITION}" \
  "${extra_args[@]}" \
  "window_index.cache_path=${INDEX}" \
  "train.steps=${STEPS}" \
  "train.batch_size=${BATCH_SIZE}" \
  "train.checkpoint_every=${CHECKPOINT_EVERY}" \
  "train.eval_every=${EVAL_EVERY}" \
  "train.eval_batches=${EVAL_BATCHES}" \
  "train.eval_windows_per_dataset=${EVAL_WINDOWS_PER_DATASET}" \
  "wandb.experiment=${JOBTAG}" \
  "output_dir=${run_dir}" \
  "hydra.run.dir=${run_dir}"

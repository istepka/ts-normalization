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

# MOMENT masked-reconstruction or TimesFM-1.0 (reduced config) forecasting
# pretraining on GiftEvalPretrain: Phases 3-7 of
# notes/05-timesfm-pretraining-loss-space-plan.md. A 6-task Slurm array, one
# GPU and one (condition, experiment_kind[, scale_assignment]) run per task,
# so all 6 runs execute in parallel (subject to GPU availability) instead of
# sequentially, and a single task failing doesn't take the other 5 down with
# it. Requires MODEL=moment|timesfm. Prefer scripts/submit_pretraining.sh
# over submitting this directly -- it wires up MODEL, the prerequisite index
# build, and the dependent aggregation job for you. For MODEL=timesfm, each
# task independently runs the architecture parity/smoke test suite first --
# "Do not begin TimesFM pretraining until the parity test passes" is an
# explicit plan instruction, so a task refuses to proceed if it fails.
#
# Requires a pre-built window index (INDEX below) -- unlike the old
# single-job version, this does NOT build it inline as a fallback: 6 parallel
# tasks racing to build and write the same cache file would be a correctness
# risk, not just wasted work. Build it once first:
#   sbatch scripts/build_gifteval_window_index.sbatch
# (scripts/submit_pretraining.sh does this automatically if missing.)
#
# Tuned against the real corpus on this cluster's H200 (see
# notes/agentic_logs): both models plateau in windows/sec well before 85%
# VRAM utilization -- pushing batch size further trades away gradient-update
# frequency for a shrinking throughput return and risks OOM. BATCH_SIZE
# below defaults to MOMENT's measured sweet spot; override with BATCH_SIZE=<n>
# (scripts/submit_pretraining.sh defaults it higher for MODEL=timesfm).
# TimesFM's window_index prediction_length must stay 128 to match every
# TimesFM config's horizon_len; only context_length is meant to be tuned.

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

if [ -z "${MODEL:-}" ]; then
  echo "ERROR: MODEL must be set to moment or timesfm" >&2
  exit 1
fi

if [ "${MODEL}" = "timesfm" ]; then
  echo "running TimesFM architecture parity/smoke tests before pretraining ..."
  uv run python -m pytest tests/test_timesfm_model.py -q
fi

CONTEXT_LENGTH=${CONTEXT_LENGTH:-512}
PREDICTION_LENGTH=${PREDICTION_LENGTH:-128}
INDEX=${INDEX:-outputs/gifteval_window_index/context${CONTEXT_LENGTH}_pred${PREDICTION_LENGTH}.parquet}
STEPS=${STEPS:-30000}
BATCH_SIZE=${BATCH_SIZE:-512}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-6000}
EVAL_EVERY=${EVAL_EVERY:-250}
EVAL_BATCHES=${EVAL_BATCHES:-50}
EVAL_WINDOWS_PER_DATASET=${EVAL_WINDOWS_PER_DATASET:-64}
SEED=${SEED:-0}
CONFIG_SIZE=${CONFIG_SIZE:-70m}
JOBTAG=${JOBTAG:-gifteval_${MODEL}_${SLURM_ARRAY_JOB_ID}}

if [ ! -f "${INDEX}" ]; then
  echo "ERROR: window index not found at ${INDEX}." >&2
  echo "Build it first: sbatch scripts/build_gifteval_window_index.sbatch" >&2
  echo "(or submit through scripts/submit_pretraining.sh, which does this for you)" >&2
  exit 1
fi

# task -> (condition, experiment_kind, scale_assignment) mapping.
if [ "${MODEL}" = "moment" ]; then
  RUNS=(
    "moment_normalized natural_mixture -"
    "moment_original natural_mixture -"
    "moment_normalized controlled_scale A"
    "moment_normalized controlled_scale B"
    "moment_original controlled_scale A"
    "moment_original controlled_scale B"
  )
else
  RUNS=(
    "timesfm_native_original natural_mixture -"
    "timesfm_normalized natural_mixture -"
    "timesfm_native_original controlled_scale A"
    "timesfm_native_original controlled_scale B"
    "timesfm_normalized controlled_scale A"
    "timesfm_normalized controlled_scale B"
  )
fi
read -r CONDITION KIND ASSIGNMENT <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

if [ "${KIND}" = "controlled_scale" ]; then
  run_dir="outputs/${JOBTAG}_scale_${CONDITION}_${ASSIGNMENT}"
  extra_args=(experiment_kind=controlled_scale "scale_assignment=${ASSIGNMENT}")
else
  run_dir="outputs/${JOBTAG}_natural_${CONDITION}"
  extra_args=(experiment_kind=natural_mixture)
fi

if [ "${MODEL}" = "timesfm" ]; then
  extra_args+=("timesfm.config_size=${CONFIG_SIZE}")
else
  extra_args+=("seed=${SEED}" "train.schedule_seed=${SEED}")
fi

echo "model=${MODEL} task ${SLURM_ARRAY_TASK_ID}: condition=${CONDITION} kind=${KIND} assignment=${ASSIGNMENT} -> ${run_dir}"

uv run python -m src.tsfm_pretraining.train --config-name="tsfm_${MODEL}" \
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
  "output_dir=${run_dir}"

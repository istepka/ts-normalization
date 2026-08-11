#!/usr/bin/env bash
#SBATCH --job-name=gifteval_timesfm
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out

# TimesFM-1.0 (reduced config) forecasting pretraining on GiftEvalPretrain:
# Phases 5-7 of notes/05-timesfm-pretraining-loss-space-plan.md. A 6-task
# Slurm array, one GPU and one (condition, experiment_kind[,
# scale_assignment]) run per task, so all 6 runs execute in parallel (subject
# to GPU availability) instead of sequentially, and a single task failing
# doesn't take the other 5 down with it. Each task independently runs the
# architecture parity/smoke test suite first -- "Do not begin TimesFM
# pretraining until the parity test passes" is an explicit plan instruction,
# so a task refuses to proceed if it fails. Prefer
# scripts/submit_timesfm_pretraining.sh over submitting this directly -- it
# wires up the prerequisite index build and the dependent aggregation job.
#
# Requires a pre-built window index (INDEX below) -- unlike the old
# single-job version, this does NOT build it inline as a fallback: 6 parallel
# tasks racing to build and write the same cache file would be a correctness
# risk, not just wasted work. Build it once first:
#   sbatch scripts/build_gifteval_window_index.sbatch
# (scripts/submit_timesfm_pretraining.sh does this automatically if missing.)
#
# CONFIG_SIZE defaults to 70m (the primary-run config per the plan: "The 70M
# configuration is the primary run if throughput permits"). Tuned against the
# real corpus on this cluster's H200 (see notes/agentic_logs): throughput
# plateaus well before 85% VRAM utilization -- pushing batch size further
# trades away gradient-update frequency for a shrinking throughput return and
# eventually OOMs. BATCH_SIZE below defaults to the measured sweet spot;
# override with BATCH_SIZE=<n> if you want to re-tune it. window_index
# prediction_length must stay 128 to match every TimesFM config's
# horizon_len; only context_length is meant to be tuned.

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

echo "running TimesFM architecture parity/smoke tests before pretraining ..."
uv run python -m pytest tests/test_timesfm_model.py -q

CONFIG_SIZE=${CONFIG_SIZE:-70m}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-512}
PREDICTION_LENGTH=128
INDEX=${INDEX:-outputs/gifteval_window_index/context${CONTEXT_LENGTH}_pred${PREDICTION_LENGTH}.parquet}
STEPS=${STEPS:-30000}
BATCH_SIZE=${BATCH_SIZE:-512}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-6000}
EVAL_EVERY=${EVAL_EVERY:-250}
EVAL_BATCHES=${EVAL_BATCHES:-50}
EVAL_WINDOWS_PER_DATASET=${EVAL_WINDOWS_PER_DATASET:-64}
JOBTAG=${JOBTAG:-gifteval_timesfm_${SLURM_ARRAY_JOB_ID}}

if [ ! -f "${INDEX}" ]; then
  echo "ERROR: window index not found at ${INDEX}." >&2
  echo "Build it first: sbatch scripts/build_gifteval_window_index.sbatch" >&2
  echo "(or submit through scripts/submit_timesfm_pretraining.sh, which does this for you)" >&2
  exit 1
fi

# task -> (condition, experiment_kind, scale_assignment) mapping.
RUNS=(
  "timesfm_native_original natural_mixture -"
  "timesfm_normalized natural_mixture -"
  "timesfm_native_original controlled_scale A"
  "timesfm_native_original controlled_scale B"
  "timesfm_normalized controlled_scale A"
  "timesfm_normalized controlled_scale B"
)
read -r CONDITION KIND ASSIGNMENT <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

if [ "${KIND}" = "controlled_scale" ]; then
  run_dir="outputs/${JOBTAG}_scale_${CONDITION}_${ASSIGNMENT}"
  extra_args=(experiment_kind=controlled_scale "scale_assignment=${ASSIGNMENT}")
else
  run_dir="outputs/${JOBTAG}_natural_${CONDITION}"
  extra_args=(experiment_kind=natural_mixture)
fi

echo "task ${SLURM_ARRAY_TASK_ID}: condition=${CONDITION} kind=${KIND} assignment=${ASSIGNMENT} -> ${run_dir}"

uv run python -m src.tsfm_pretraining.train --config-name=tsfm_timesfm \
  device=cuda \
  condition="${CONDITION}" \
  "${extra_args[@]}" \
  "timesfm.config_size=${CONFIG_SIZE}" \
  "window_index.cache_path=${INDEX}" \
  "train.steps=${STEPS}" \
  "train.batch_size=${BATCH_SIZE}" \
  "train.checkpoint_every=${CHECKPOINT_EVERY}" \
  "train.eval_every=${EVAL_EVERY}" \
  "train.eval_batches=${EVAL_BATCHES}" \
  "train.eval_windows_per_dataset=${EVAL_WINDOWS_PER_DATASET}" \
  "wandb.experiment=${JOBTAG}" \
  "output_dir=${run_dir}"

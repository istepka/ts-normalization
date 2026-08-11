#!/usr/bin/env bash
#SBATCH --job-name=gifteval_moment
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out

# MOMENT masked-reconstruction pretraining on GiftEvalPretrain: Phases 3-4 of
# notes/05-timesfm-pretraining-loss-space-plan.md. A 6-task Slurm array, one
# GPU and one (condition, experiment_kind[, scale_assignment]) run per task,
# so all 6 runs execute in parallel (subject to GPU availability) instead of
# sequentially, and a single task failing doesn't take the other 5 down with
# it. Prefer scripts/submit_moment_pretraining.sh over submitting this
# directly -- it wires up the prerequisite index build and the dependent
# aggregation job for you.
#
# Requires a pre-built window index (INDEX below) -- unlike the old
# single-job version, this does NOT build it inline as a fallback: 6 parallel
# tasks racing to build and write the same cache file would be a correctness
# risk, not just wasted work. Build it once first:
#   sbatch scripts/build_gifteval_window_index.sbatch
# (scripts/submit_moment_pretraining.sh does this automatically if missing.)
#
# Tuned against the real corpus on this cluster's H200 (see
# notes/agentic_logs): MOMENT-base (~66M params, d_model=768/12 T5 layers,
# the size conf/tsfm_moment.yaml now defaults to) plateaus in windows/sec
# well before 85% VRAM utilization -- pushing batch size further trades away
# gradient-update frequency for a shrinking throughput return and risks OOM.
# BATCH_SIZE below defaults to the measured sweet spot; override with
# BATCH_SIZE=<n> if you want to re-tune it.

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

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
JOBTAG=${JOBTAG:-gifteval_moment_${SLURM_ARRAY_JOB_ID}}

if [ ! -f "${INDEX}" ]; then
  echo "ERROR: window index not found at ${INDEX}." >&2
  echo "Build it first: sbatch scripts/build_gifteval_window_index.sbatch" >&2
  echo "(or submit through scripts/submit_moment_pretraining.sh, which does this for you)" >&2
  exit 1
fi

# task -> (condition, experiment_kind, scale_assignment) mapping.
RUNS=(
  "moment_normalized natural_mixture -"
  "moment_original natural_mixture -"
  "moment_normalized controlled_scale A"
  "moment_normalized controlled_scale B"
  "moment_original controlled_scale A"
  "moment_original controlled_scale B"
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

uv run python -m src.tsfm_pretraining.train --config-name=tsfm_moment \
  device=cuda \
  condition="${CONDITION}" \
  "${extra_args[@]}" \
  "seed=${SEED}" \
  "window_index.cache_path=${INDEX}" \
  "train.steps=${STEPS}" \
  "train.batch_size=${BATCH_SIZE}" \
  "train.schedule_seed=${SEED}" \
  "train.checkpoint_every=${CHECKPOINT_EVERY}" \
  "train.eval_every=${EVAL_EVERY}" \
  "train.eval_batches=${EVAL_BATCHES}" \
  "train.eval_windows_per_dataset=${EVAL_WINDOWS_PER_DATASET}" \
  "wandb.experiment=${JOBTAG}" \
  "output_dir=${run_dir}"

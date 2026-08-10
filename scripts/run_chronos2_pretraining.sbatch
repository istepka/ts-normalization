#!/usr/bin/env bash
#SBATCH --job-name=gifteval_chronos2
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-23%8
#SBATCH --output=logs/%x-%A_%a.out

# Runs the Chronos-2 loss-space study with four seeds. Each seed has two
# natural-scale runs and four controlled-scale runs. The array concurrency is
# capped at eight GPU jobs. This script must be submitted through Slurm and
# must never be executed on the Rhea login node.

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

export CUBLAS_WORKSPACE_CONFIG=:4096:8

INDEX=${INDEX:-outputs/gifteval_window_index/context512_pred128.parquet}
STEPS=${STEPS:-30000}
BATCH_SIZE=${BATCH_SIZE:-128}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-2000}
EVAL_EVERY=${EVAL_EVERY:-250}
EVAL_BATCHES=${EVAL_BATCHES:-50}
EVAL_WINDOWS_PER_DATASET=${EVAL_WINDOWS_PER_DATASET:-64}
SEEDS_CSV=${SEEDS_CSV:-0,1,2,3}
JOBTAG=${JOBTAG:-gifteval_chronos2_${SLURM_ARRAY_JOB_ID}}

if [ ! -f "${INDEX}" ]; then
  echo "ERROR: window index not found at ${INDEX}" >&2
  exit 1
fi

IFS=, read -r -a SEEDS <<< "${SEEDS_CSV}"
RUNS=(
  "chronos2_normalized natural_mixture natural"
  "chronos2_original natural_mixture natural"
  "chronos2_normalized controlled_scale A"
  "chronos2_normalized controlled_scale B"
  "chronos2_original controlled_scale A"
  "chronos2_original controlled_scale B"
)

seed_index=$((SLURM_ARRAY_TASK_ID / ${#RUNS[@]}))
run_index=$((SLURM_ARRAY_TASK_ID % ${#RUNS[@]}))
SEED=${SEEDS[$seed_index]}
read -r CONDITION KIND ASSIGNMENT <<< "${RUNS[$run_index]}"

if [ "${KIND}" = "controlled_scale" ]; then
  experiment_args=(experiment_kind=controlled_scale "scale_assignment=${ASSIGNMENT}")
else
  experiment_args=(experiment_kind=natural_mixture scale_assignment=null)
fi
run_dir="outputs/${JOBTAG}_seed${SEED}_${CONDITION}_${ASSIGNMENT}"

echo "seed=${SEED} condition=${CONDITION} kind=${KIND} assignment=${ASSIGNMENT}"

PYTHONPATH=. uv run python -m src.tsfm_pretraining.train \
  --config-name=tsfm_chronos2 \
  device=cuda \
  condition="${CONDITION}" \
  "${experiment_args[@]}" \
  seed="${SEED}" \
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

#!/usr/bin/env bash
#SBATCH --job-name=moment_natural
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%6
#SBATCH --output=logs/%x-%A_%a.out

# Reruns MOMENT natural-scale seeds 0 through 3 after applying safe gradient
# rescaling to the MOMENT training step. Every experiment value is encoded
# in this file.

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

INDEX=outputs/gifteval_window_index/context512_pred128.parquet
STEPS=30000
BATCH_SIZE=512
CHECKPOINT_EVERY=30000
EVAL_EVERY=250
EVAL_BATCHES=50
EVAL_WINDOWS_PER_DATASET=64

CONDITIONS=(moment_normalized moment_original)

seed_index=$((SLURM_ARRAY_TASK_ID / ${#CONDITIONS[@]}))
condition_index=$((SLURM_ARRAY_TASK_ID % ${#CONDITIONS[@]}))
SEED=$seed_index
CONDITION=${CONDITIONS[$condition_index]}

JOBTAG=gifteval_moment_${SLURM_ARRAY_JOB_ID}
run_dir="outputs/${JOBTAG}_seed${SEED}_${CONDITION}_natural"

uv run python -m src.tsfm_pretraining.train --config-name=tsfm_moment \
  device=cuda \
  condition="${CONDITION}" \
  experiment_kind=natural_mixture \
  scale_assignment=null \
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

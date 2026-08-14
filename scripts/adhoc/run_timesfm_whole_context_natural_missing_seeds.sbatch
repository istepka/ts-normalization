#!/usr/bin/env bash
#SBATCH --job-name=timesfm_natural
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/%x-%A_%a.out

# Reruns whole-context TimesFM natural-scale seeds 0 and 1 with safe gradient
# rescaling enabled. Every experiment value is encoded in this file.

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

export CUBLAS_WORKSPACE_CONFIG=:4096:8

INDEX=outputs/gifteval_window_index/context512_pred128.parquet
CONDITIONS=(timesfm_native_original timesfm_normalized)
seed_index=$((SLURM_ARRAY_TASK_ID / ${#CONDITIONS[@]}))
condition_index=$((SLURM_ARRAY_TASK_ID % ${#CONDITIONS[@]}))
SEED=$seed_index
CONDITION=${CONDITIONS[$condition_index]}
JOBTAG=timesfm_whole_context_natural_${SLURM_ARRAY_JOB_ID}
run_dir="outputs/${JOBTAG}_seed${SEED}_${CONDITION}"

uv run python -m src.training.tsfm --config-name=tsfm_timesfm \
  device=cuda \
  condition="${CONDITION}" \
  experiment_kind=natural_mixture \
  scale_assignment=null \
  seed="${SEED}" \
  timesfm.config_size=70m \
  timesfm.normalization_mode=whole_context \
  timesfm.objective=mse \
  "window_index.cache_path=${INDEX}" \
  train.steps=30000 \
  train.batch_size=512 \
  "train.schedule_seed=${SEED}" \
  train.checkpoint_every=30000 \
  train.eval_every=2000 \
  train.eval_batches=8 \
  train.eval_windows_per_dataset=32 \
  "wandb.experiment=${JOBTAG}" \
  "output_dir=${run_dir}"

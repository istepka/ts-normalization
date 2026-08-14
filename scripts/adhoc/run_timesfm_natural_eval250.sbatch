#!/usr/bin/env bash
#SBATCH --job-name=timesfm_eval250
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs
source scripts/output_paths.sh

export CUBLAS_WORKSPACE_CONFIG=:4096:8

index=outputs/gifteval_window_index/context512_pred128.parquet
conditions=(timesfm_native_original timesfm_normalized)
seed=$((SLURM_ARRAY_TASK_ID / ${#conditions[@]}))
condition_index=$((SLURM_ARRAY_TASK_ID % ${#conditions[@]}))
condition=${conditions[$condition_index]}
job_tag=timesfm_natural_eval250_${SLURM_ARRAY_JOB_ID}
run_root=$(output_path experiments "tsfm_pretraining/timesfm" "$job_tag")
run_dir="${run_root}/seed${seed}_${condition}_natural"

PYTHONPATH=. uv run python -m src.training.tsfm \
  --config-name=tsfm_timesfm \
  device=cuda \
  condition="$condition" \
  experiment_kind=natural_mixture \
  scale_assignment=null \
  seed="$seed" \
  timesfm.config_size=70m \
  timesfm.normalization_mode=whole_context \
  timesfm.objective=mse \
  "window_index.cache_path=${index}" \
  train.steps=30000 \
  train.batch_size=512 \
  "train.schedule_seed=${seed}" \
  train.checkpoint_every=30000 \
  train.eval_every=250 \
  train.eval_batches=8 \
  train.eval_windows_per_dataset=32 \
  "wandb.experiment=${job_tag}" \
  "output_dir=${run_dir}" \
  "hydra.run.dir=${run_dir}"

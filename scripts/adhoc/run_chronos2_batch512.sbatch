#!/usr/bin/env bash
#SBATCH --job-name=chronos2_b512
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-23%8
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

export CUBLAS_WORKSPACE_CONFIG=:4096:8

INDEX=outputs/gifteval_window_index/context512_pred128.parquet
RUNS=(
  "chronos2_normalized natural_mixture natural"
  "chronos2_original natural_mixture natural"
  "chronos2_normalized controlled_scale A"
  "chronos2_normalized controlled_scale B"
  "chronos2_original controlled_scale A"
  "chronos2_original controlled_scale B"
)

seed=$((SLURM_ARRAY_TASK_ID / ${#RUNS[@]}))
run_index=$((SLURM_ARRAY_TASK_ID % ${#RUNS[@]}))
read -r condition kind assignment <<< "${RUNS[$run_index]}"

if [[ $kind == controlled_scale ]]; then
  experiment_args=(experiment_kind=controlled_scale "scale_assignment=${assignment}")
else
  experiment_args=(experiment_kind=natural_mixture scale_assignment=null)
fi

job_tag=gifteval_chronos2_b512_${SLURM_ARRAY_JOB_ID}
run_dir="outputs/${job_tag}_seed${seed}_${condition}_${assignment}"

PYTHONPATH=. uv run python -m src.tsfm_pretraining.train \
  --config-name=tsfm_chronos2 \
  device=cuda \
  condition="$condition" \
  "${experiment_args[@]}" \
  seed="$seed" \
  "window_index.cache_path=${INDEX}" \
  train.steps=30000 \
  train.schedule_steps=null \
  train.batch_size=512 \
  "train.schedule_seed=${seed}" \
  train.checkpoint_every=6000 \
  train.eval_every=250 \
  train.eval_batches=50 \
  train.eval_windows_per_dataset=64 \
  "wandb.experiment=${job_tag}" \
  "output_dir=${run_dir}"

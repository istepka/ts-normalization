#!/usr/bin/env bash
#SBATCH --job-name=tsfm_continue60k
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --array=0-31%8
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs
source scripts/output_paths.sh

export CUBLAS_WORKSPACE_CONFIG=:4096:8

index=outputs/gifteval_window_index/context512_pred128.parquet
models=(moment moment timesfm timesfm chronos2 chronos2 moirai2 moirai2)
conditions=(
  moment_normalized
  moment_original
  timesfm_normalized
  timesfm_native_original
  chronos2_normalized
  chronos2_original
  moirai2_normalized
  moirai2_original
)

seed=$((SLURM_ARRAY_TASK_ID / ${#models[@]}))
variant_index=$((SLURM_ARRAY_TASK_ID % ${#models[@]}))
model=${models[$variant_index]}
condition=${conditions[$variant_index]}

case $model in
  moment)
    config_name=tsfm_moment
    source_dir="outputs/2026-08-11/experiments/legacy_runs/gifteval_moment_28827_seed${seed}_${condition}_natural"
    model_args=()
    eval_batches=50
    eval_windows_per_dataset=64
    ;;
  timesfm)
    config_name=tsfm_timesfm
    source_dir="outputs/2026-08-13/experiments/tsfm_pretraining/timesfm/timesfm_natural_eval250_30192/seed${seed}_${condition}_natural"
    model_args=(
      timesfm.config_size=70m
      timesfm.normalization_mode=whole_context
      timesfm.objective=mse
    )
    eval_batches=8
    eval_windows_per_dataset=32
    ;;
  chronos2)
    config_name=tsfm_chronos2
    source_dir="outputs/2026-08-12/experiments/tsfm_pretraining/chronos2/gifteval_chronos2_b512_29436/seed${seed}_${condition}_natural"
    model_args=()
    eval_batches=50
    eval_windows_per_dataset=64
    ;;
  moirai2)
    config_name=tsfm_moirai2
    source_dir="outputs/2026-08-12/experiments/tsfm_pretraining/moirai2/gifteval_moirai2_b512_29437/seed${seed}_${condition}_natural"
    model_args=()
    eval_batches=50
    eval_windows_per_dataset=64
    ;;
esac

checkpoint="${source_dir}/checkpoint_step30000.pt"
job_tag=tsfm_natural_continue60k_${SLURM_ARRAY_JOB_ID}
run_root=$(output_path experiments "tsfm_pretraining/${model}" "$job_tag")
run_dir="${run_root}/seed${seed}_${condition}_natural"

PYTHONPATH=. uv run python -m src.training.tsfm \
  --config-name="$config_name" \
  device=cuda \
  condition="$condition" \
  experiment_kind=natural_mixture \
  scale_assignment=null \
  seed="$seed" \
  "${model_args[@]}" \
  "window_index.cache_path=${index}" \
  train.steps=60000 \
  train.schedule_steps=null \
  train.batch_size=512 \
  "train.schedule_seed=${seed}" \
  train.checkpoint_every=6000 \
  train.eval_every=250 \
  "train.eval_batches=${eval_batches}" \
  "train.eval_windows_per_dataset=${eval_windows_per_dataset}" \
  "train.resume_from=${checkpoint}" \
  "wandb.experiment=${job_tag}" \
  "output_dir=${run_dir}" \
  "hydra.run.dir=${run_dir}"

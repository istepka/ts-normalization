#!/usr/bin/env bash
#SBATCH --job-name=loss_early_sanity
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-39%8
#SBATCH --output=logs/%x-%A_%a.out

# One-seed early-training sanity check for four TSFM architectures. Each task
# starts from seed 0 and stops at one requested step, so paired loss conditions
# share initialization, batch order, and evaluation windows.

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

export CUBLAS_WORKSPACE_CONFIG=:4096:8

INDEX=outputs/gifteval_window_index/context512_pred128.parquet
STEPS=(1 10 50 100 250)

model_index=$((SLURM_ARRAY_TASK_ID / 10))
remainder=$((SLURM_ARRAY_TASK_ID % 10))
condition_index=$((remainder / 5))
step_index=$((remainder % 5))
STEP=${STEPS[$step_index]}

case $model_index in
  0)
    MODEL=moment
    CONFIG=tsfm_moment
    CONDITIONS=(moment_normalized moment_original)
    BATCH_SIZE=512
    ;;
  1)
    MODEL=timesfm
    CONFIG=tsfm_timesfm
    CONDITIONS=(timesfm_normalized timesfm_native_original)
    BATCH_SIZE=512
    ;;
  2)
    MODEL=chronos2
    CONFIG=tsfm_chronos2
    CONDITIONS=(chronos2_normalized chronos2_original)
    BATCH_SIZE=128
    ;;
  3)
    MODEL=moirai2
    CONFIG=tsfm_moirai2
    CONDITIONS=(moirai2_normalized moirai2_original)
    BATCH_SIZE=128
    ;;
esac

CONDITION=${CONDITIONS[$condition_index]}
JOBTAG=loss_early_sanity_${SLURM_ARRAY_JOB_ID}
run_dir="outputs/${JOBTAG}_${MODEL}_${CONDITION}_step${STEP}"

extra_args=()
if [[ $MODEL == timesfm ]]; then
  extra_args+=(
    timesfm.config_size=70m
    timesfm.normalization_mode=whole_context
    timesfm.objective=mse
  )
fi

uv run python -m src.training.tsfm --config-name="$CONFIG" \
  device=cuda \
  condition="$CONDITION" \
  experiment_kind=natural_mixture \
  scale_assignment=null \
  seed=0 \
  "window_index.cache_path=${INDEX}" \
  "train.steps=${STEP}" \
  train.schedule_steps=30000 \
  "train.batch_size=${BATCH_SIZE}" \
  train.schedule_seed=0 \
  "train.eval_every=${STEP}" \
  train.eval_batches=50 \
  train.eval_windows_per_dataset=64 \
  "train.checkpoint_every=${STEP}" \
  wandb.mode=disabled \
  "wandb.experiment=${JOBTAG}" \
  "output_dir=${run_dir}" \
  "${extra_args[@]}"

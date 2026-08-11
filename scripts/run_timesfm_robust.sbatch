#!/usr/bin/env bash
#SBATCH --job-name=timesfm_robust
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

export CUBLAS_WORKSPACE_CONFIG=:4096:8

CONFIG_SIZE=${CONFIG_SIZE:-17m}
INDEX=${INDEX:-outputs/gifteval_window_index/context512_pred128.parquet}
STEPS=${STEPS:-2000}
BATCH_SIZE=${BATCH_SIZE:-512}
EVAL_EVERY=${EVAL_EVERY:-250}
EVAL_BATCHES=${EVAL_BATCHES:-8}
EVAL_WINDOWS_PER_DATASET=${EVAL_WINDOWS_PER_DATASET:-32}
OBJECTIVE=${OBJECTIVE:-mse}
SEEDS_CSV=${SEEDS_CSV:-0,1,2,3}
MODES_CSV=${MODES_CSV:-whole_context}
EXPERIMENT_KIND=${EXPERIMENT_KIND:-controlled_scale}
JOBTAG=${JOBTAG:-timesfm_robust_${SLURM_ARRAY_JOB_ID}}

if [ ! -f "${INDEX}" ]; then
  echo "ERROR: window index not found at ${INDEX}" >&2
  exit 1
fi

IFS=, read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=, read -r -a MODES <<< "${MODES_CSV}"
CONDITIONS=(timesfm_native_original timesfm_normalized)
if [ "${EXPERIMENT_KIND}" = "controlled_scale" ]; then
  ASSIGNMENTS=(A B)
elif [ "${EXPERIMENT_KIND}" = "natural_mixture" ]; then
  ASSIGNMENTS=(natural)
else
  echo "ERROR: unsupported experiment kind ${EXPERIMENT_KIND}" >&2
  exit 1
fi

assignment_index=$((SLURM_ARRAY_TASK_ID % ${#ASSIGNMENTS[@]}))
condition_index=$(((SLURM_ARRAY_TASK_ID / ${#ASSIGNMENTS[@]}) % ${#CONDITIONS[@]}))
mode_index=$(((SLURM_ARRAY_TASK_ID / (${#ASSIGNMENTS[@]} * ${#CONDITIONS[@]})) % ${#MODES[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID / (${#ASSIGNMENTS[@]} * ${#CONDITIONS[@]} * ${#MODES[@]})))

SEED=${SEEDS[$seed_index]}
MODE=${MODES[$mode_index]}
CONDITION=${CONDITIONS[$condition_index]}
ASSIGNMENT=${ASSIGNMENTS[$assignment_index]}
run_dir="outputs/${JOBTAG}_${MODE}_${OBJECTIVE}_seed${SEED}_${CONDITION}_${ASSIGNMENT}"

echo "task=${SLURM_ARRAY_TASK_ID} seed=${SEED} mode=${MODE} objective=${OBJECTIVE} condition=${CONDITION} assignment=${ASSIGNMENT} kind=${EXPERIMENT_KIND}"

if [ "${EXPERIMENT_KIND}" = "controlled_scale" ]; then
  experiment_args=(experiment_kind=controlled_scale "scale_assignment=${ASSIGNMENT}")
else
  experiment_args=(experiment_kind=natural_mixture scale_assignment=null)
fi

uv run python -m src.tsfm_pretraining.train --config-name=tsfm_timesfm \
  device=cuda \
  condition="${CONDITION}" \
  "${experiment_args[@]}" \
  seed="${SEED}" \
  "timesfm.config_size=${CONFIG_SIZE}" \
  "timesfm.normalization_mode=${MODE}" \
  "timesfm.objective=${OBJECTIVE}" \
  "window_index.cache_path=${INDEX}" \
  "train.steps=${STEPS}" \
  "train.batch_size=${BATCH_SIZE}" \
  "train.schedule_seed=${SEED}" \
  "train.checkpoint_every=${STEPS}" \
  "train.eval_every=${EVAL_EVERY}" \
  "train.eval_batches=${EVAL_BATCHES}" \
  "train.eval_windows_per_dataset=${EVAL_WINDOWS_PER_DATASET}" \
  "wandb.experiment=${JOBTAG}" \
  "output_dir=${run_dir}"

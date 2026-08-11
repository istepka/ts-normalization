#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export CONFIG_SIZE=${CONFIG_SIZE:-17m}
export INDEX=${INDEX:-outputs/gifteval_window_index/context512_pred128.parquet}
export STEPS=${STEPS:-2000}
export BATCH_SIZE=${BATCH_SIZE:-512}
export EVAL_EVERY=${EVAL_EVERY:-250}
export EVAL_BATCHES=${EVAL_BATCHES:-8}
export EVAL_WINDOWS_PER_DATASET=${EVAL_WINDOWS_PER_DATASET:-32}
export OBJECTIVE=${OBJECTIVE:-mse}
export SEEDS_CSV=${SEEDS_CSV:-0,1,2,3}
export MODES_CSV=${MODES_CSV:-whole_context}
export EXPERIMENT_KIND=${EXPERIMENT_KIND:-controlled_scale}

if [ ! -f "${INDEX}" ]; then
  echo "ERROR: window index not found at ${INDEX}" >&2
  exit 1
fi

IFS=, read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=, read -r -a MODES <<< "${MODES_CSV}"
if [ "${EXPERIMENT_KIND}" = "controlled_scale" ]; then
  assignments=2
elif [ "${EXPERIMENT_KIND}" = "natural_mixture" ]; then
  assignments=1
else
  echo "ERROR: unsupported experiment kind ${EXPERIMENT_KIND}" >&2
  exit 1
fi
tasks=$((${#SEEDS[@]} * ${#MODES[@]} * 2 * assignments))
last_task=$((tasks - 1))

ARRAY_JOB=$(sbatch --parsable --array="0-${last_task}" \
  scripts/run_timesfm_robust.sbatch)
export JOBTAG="timesfm_robust_${ARRAY_JOB}"

echo "array job ${ARRAY_JOB}"
echo "tasks ${tasks}"
echo "config ${CONFIG_SIZE}"
echo "steps ${STEPS}"
echo "seeds ${SEEDS_CSV}"
echo "normalization modes ${MODES_CSV}"
echo "objective ${OBJECTIVE}"
echo "experiment kind ${EXPERIMENT_KIND}"

#!/usr/bin/env bash

# Submits the four-seed Moirai-2.0 study. This script only calls Slurm. The
# training work runs on allocated compute nodes.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export INDEX=${INDEX:-outputs/gifteval_window_index/context512_pred128.parquet}
export SEEDS_CSV=${SEEDS_CSV:-0,1,2,3}

if [ ! -f "${INDEX}" ]; then
  echo "ERROR: window index not found at ${INDEX}" >&2
  exit 1
fi

IFS=, read -r -a SEEDS <<< "${SEEDS_CSV}"
tasks=$((${#SEEDS[@]} * 6))
last_task=$((tasks - 1))

ARRAY_JOB=$(sbatch --parsable --array="0-${last_task}%8" \
  scripts/run_moirai2_pretraining.sbatch)

echo "array job ${ARRAY_JOB}"
echo "tasks ${tasks}"
echo "seeds ${SEEDS_CSV}"
echo "maximum concurrent GPU jobs 8"
echo "no fifth seed is included"

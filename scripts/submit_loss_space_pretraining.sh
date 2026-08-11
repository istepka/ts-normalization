#!/usr/bin/env bash

# Submits the four-seed loss-space study for MODEL=chronos2 or MODEL=moirai2.
# This script only calls Slurm. The training work runs on allocated compute
# nodes.
#
# Usage:
#   MODEL=chronos2 scripts/submit_loss_space_pretraining.sh
#   MODEL=moirai2 scripts/submit_loss_space_pretraining.sh

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

case "${MODEL:-}" in
  chronos2 | moirai2) ;;
  *)
    echo "ERROR: MODEL must be set to chronos2 or moirai2" >&2
    exit 1
    ;;
esac
export MODEL

export INDEX=${INDEX:-outputs/gifteval_window_index/context512_pred128.parquet}
export SEEDS_CSV=${SEEDS_CSV:-0,1,2,3}

if [ ! -f "${INDEX}" ]; then
  echo "ERROR: window index not found at ${INDEX}" >&2
  exit 1
fi

IFS=, read -r -a SEEDS <<< "${SEEDS_CSV}"
tasks=$((${#SEEDS[@]} * 6))
last_task=$((tasks - 1))

ARRAY_JOB=$(sbatch --parsable --job-name="gifteval_${MODEL}" \
  --array="0-${last_task}%8" \
  scripts/run_loss_space_pretraining.sbatch)

echo "array job ${ARRAY_JOB}"
echo "model ${MODEL}"
echo "tasks ${tasks}"
echo "seeds ${SEEDS_CSV}"
echo "maximum concurrent GPU jobs 8"
echo "no fifth seed is included"

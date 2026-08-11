#!/usr/bin/env bash
# Submits the full MOMENT or TimesFM GiftEvalPretrain study as a dependency
# chain:
#   1. build_gifteval_window_index.sbatch (CPU) -- only if INDEX doesn't exist
#   2. run_pretraining.sbatch (GPU array, 6 parallel tasks)
#   3. aggregate_pretraining.sbatch (CPU) -- afterok all 6 array tasks
#
# Usage:
#   MODEL=moment scripts/submit_pretraining.sh
#   MODEL=timesfm STEPS=5000 scripts/submit_pretraining.sh   # dry run
#
# To resubmit only failed array tasks against the SAME output namespace as an
# earlier submission (so aggregation still finds all 6 run dirs), pass that
# submission's JOBTAG and the failed task indices directly:
#   MODEL=moment JOBTAG=gifteval_moment_12345 sbatch --array=3 scripts/run_pretraining.sbatch
#   MODEL=moment JOBTAG=gifteval_moment_12345 sbatch scripts/aggregate_pretraining.sbatch

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

case "${MODEL:-}" in
  moment) default_batch_size=512 ;;
  timesfm) default_batch_size=1024 ;;
  *)
    echo "ERROR: MODEL must be set to moment or timesfm" >&2
    exit 1
    ;;
esac
export MODEL

export CONTEXT_LENGTH=${CONTEXT_LENGTH:-512}
if [ "${MODEL}" = "timesfm" ]; then
  export PREDICTION_LENGTH=128
else
  export PREDICTION_LENGTH=${PREDICTION_LENGTH:-128}
fi
export INDEX=${INDEX:-outputs/gifteval_window_index/context${CONTEXT_LENGTH}_pred${PREDICTION_LENGTH}.parquet}
export STEPS=${STEPS:-30000}
export BATCH_SIZE=${BATCH_SIZE:-$default_batch_size}
export CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-6000}
export EVAL_EVERY=${EVAL_EVERY:-250}
export EVAL_BATCHES=${EVAL_BATCHES:-50}
export EVAL_WINDOWS_PER_DATASET=${EVAL_WINDOWS_PER_DATASET:-64}
export CONFIG_SIZE=${CONFIG_SIZE:-70m}
export OUTPUT="${INDEX}"

DEPENDENCY_ARGS=()
if [ ! -f "${INDEX}" ]; then
  echo "index not found at ${INDEX}, submitting build job first"
  BUILD_JOB=$(sbatch --parsable scripts/build_gifteval_window_index.sbatch)
  echo "  build job: ${BUILD_JOB}"
  DEPENDENCY_ARGS=(--dependency="afterok:${BUILD_JOB}")
fi

ARRAY_JOB=$(sbatch --parsable --job-name="gifteval_${MODEL}" \
  "${DEPENDENCY_ARGS[@]}" scripts/run_pretraining.sbatch)
echo "array job: ${ARRAY_JOB} (6 tasks, model=${MODEL}, steps=${STEPS}, batch_size=${BATCH_SIZE}, checkpoint_every=${CHECKPOINT_EVERY})"

export JOBTAG="gifteval_${MODEL}_${ARRAY_JOB}"
AGG_JOB=$(sbatch --parsable --job-name="gifteval_${MODEL}_aggregate" \
  --dependency="afterok:${ARRAY_JOB}" \
  scripts/aggregate_pretraining.sbatch)
echo "aggregate job: ${AGG_JOB} (depends on ${ARRAY_JOB} completing ok)"
echo "outputs will land under outputs/${JOBTAG}_*"

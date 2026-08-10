#!/usr/bin/env bash
# Submits the full MOMENT GiftEvalPretrain study as a dependency chain:
#   1. build_gifteval_window_index.sbatch (CPU) -- only if INDEX doesn't exist
#   2. run_moment_pretraining.sbatch (GPU array, 6 parallel tasks)
#   3. aggregate_moment_pretraining.sbatch (CPU) -- afterok all 6 array tasks
#
# Usage:
#   scripts/submit_moment_pretraining.sh
#   STEPS=5000 BATCH_SIZE=512 scripts/submit_moment_pretraining.sh   # dry run
#
# To resubmit only failed array tasks against the SAME output namespace as an
# earlier submission (so aggregation still finds all 6 run dirs), pass that
# submission's JOBTAG and the failed task indices directly:
#   JOBTAG=gifteval_moment_12345 sbatch --array=3 scripts/run_moment_pretraining.sbatch
#   JOBTAG=gifteval_moment_12345 sbatch scripts/aggregate_moment_pretraining.sbatch

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export CONTEXT_LENGTH=${CONTEXT_LENGTH:-512}
export PREDICTION_LENGTH=${PREDICTION_LENGTH:-128}
export INDEX=${INDEX:-outputs/gifteval_window_index/context${CONTEXT_LENGTH}_pred${PREDICTION_LENGTH}.parquet}
export STEPS=${STEPS:-30000}
export BATCH_SIZE=${BATCH_SIZE:-512}
export CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-2000}
export EVAL_EVERY=${EVAL_EVERY:-250}
export EVAL_BATCHES=${EVAL_BATCHES:-50}
export EVAL_WINDOWS_PER_DATASET=${EVAL_WINDOWS_PER_DATASET:-64}
export OUTPUT="${INDEX}"

DEPENDENCY_ARGS=()
if [ ! -f "${INDEX}" ]; then
  echo "index not found at ${INDEX}, submitting build job first"
  BUILD_JOB=$(sbatch --parsable scripts/build_gifteval_window_index.sbatch)
  echo "  build job: ${BUILD_JOB}"
  DEPENDENCY_ARGS=(--dependency="afterok:${BUILD_JOB}")
fi

ARRAY_JOB=$(sbatch --parsable "${DEPENDENCY_ARGS[@]}" scripts/run_moment_pretraining.sbatch)
echo "array job: ${ARRAY_JOB} (6 tasks, steps=${STEPS}, batch_size=${BATCH_SIZE}, checkpoint_every=${CHECKPOINT_EVERY})"

export JOBTAG="gifteval_moment_${ARRAY_JOB}"
AGG_JOB=$(sbatch --parsable \
  --dependency="afterok:${ARRAY_JOB}" \
  scripts/aggregate_moment_pretraining.sbatch)
echo "aggregate job: ${AGG_JOB} (depends on ${ARRAY_JOB} completing ok)"
echo "outputs will land under outputs/${JOBTAG}_*"

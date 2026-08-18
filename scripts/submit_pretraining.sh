#!/usr/bin/env bash
# Submits a GiftEvalPretrain loss-space study for
# MODEL=moment|timesfm|chronos2|moirai2 via scripts/run_pretraining.sbatch.
#
# MOMENT/TimesFM run the full dependency chain:
#   1. build_gifteval_window_index.sbatch (CPU) -- only if INDEX doesn't exist
#   2. run_pretraining.sbatch (GPU array, 6 parallel tasks, one SEED)
#   3. aggregate_pretraining.sbatch (CPU) -- afterok all 6 array tasks
# Chronos-2/Moirai-2.0 sweep SEEDS_CSV directly (24-task array by default,
# capped at 8 concurrent GPU jobs) and are not auto-aggregated.
#
# Usage:
#   MODEL=moment scripts/submit_pretraining.sh
#   MODEL=timesfm STEPS=5000 scripts/submit_pretraining.sh   # dry run
#   MODEL=chronos2 scripts/submit_pretraining.sh
#   MODEL=moirai2 SEEDS_CSV=0,1 scripts/submit_pretraining.sh
#   # natural scale only, 3 seeds, no scale swap (6 tasks)
#   MODEL=chronos2 NATURAL_ONLY=1 SEEDS_CSV=0,1,2 scripts/submit_pretraining.sh
#
# To resubmit only failed MOMENT/TimesFM array tasks against the SAME output
# namespace as an earlier submission (so aggregation still finds all 6 run
# dirs), pass that submission's JOBTAG and the failed task indices directly:
#   MODEL=moment JOBTAG=gifteval_moment_12345 sbatch --array=3 scripts/run_pretraining.sbatch
#   MODEL=moment JOBTAG=gifteval_moment_12345 sbatch scripts/aggregate_pretraining.sbatch

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
source scripts/output_paths.sh

RUN_DATE=${RUN_DATE:-$(date +%F)}
export RUN_DATE

case "${MODEL:-}" in
  moment | timesfm | chronos2 | moirai2) ;;
  *)
    echo "ERROR: MODEL must be set to moment, timesfm, chronos2, or moirai2" >&2
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
# The window index is a shared cache, not a run artifact: every paired run must
# read the same file, so it lives at a stable path rather than a dated one.
INDEX_DIR=${INDEX_DIR:-outputs/gifteval_window_index}
export INDEX=${INDEX:-${INDEX_DIR}/context${CONTEXT_LENGTH}_pred${PREDICTION_LENGTH}_heldout.parquet}
export STEPS=${STEPS:-30000}
export BATCH_SIZE=${BATCH_SIZE:-512}
export CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-6000}
export EVAL_EVERY=${EVAL_EVERY:-250}
export EVAL_BATCHES=${EVAL_BATCHES:-50}
export EVAL_WINDOWS_PER_DATASET=${EVAL_WINDOWS_PER_DATASET:-64}
export CONFIG_SIZE=${CONFIG_SIZE:-70m}
# Passed straight through to run_pretraining.sbatch; see the comment there.
# Slurm dependency for the training array, e.g. afterok:32807 to resume a run
# that has not finished yet. The eval chain hangs off the array either way.
DEPENDENCY=${DEPENDENCY:-}
TRAIN_DEP_ARGS=()
if [ -n "${DEPENDENCY}" ]; then
  TRAIN_DEP_ARGS=(--dependency="${DEPENDENCY}")
fi
export RESUME_ROOT=${RESUME_ROOT:-}
export RESUME_STEP=${RESUME_STEP:-}
export CORPUS_ROOT=${CORPUS_ROOT:-}
export MIN_CONTEXT_LENGTH=${MIN_CONTEXT_LENGTH:-}
export MIN_PREDICTION_LENGTH=${MIN_PREDICTION_LENGTH:-}
# NATURAL_ONLY=1 keeps only the two natural_mixture runs per seed and drops the
# four controlled_scale ones, so the array is a third of its usual size.
export NATURAL_ONLY=${NATURAL_ONLY:-0}
if [ "${NATURAL_ONLY}" = "1" ]; then
  RUNS_PER_SEED=2
else
  RUNS_PER_SEED=6
fi

# Chains the held-out evaluation onto a pretraining array: one GPU task per
# run, then a CPU collector that merges them into a single report. Set EVAL=0
# to skip. MOMENT has no forecast head, so the harness does not score it.
export EVAL=${EVAL:-1}
submit_eval_chain() {
  local array_job=$1 last_task=$2
  if [ "${EVAL}" != "1" ] || [ "${MODEL}" = "moment" ]; then
    return 0
  fi
  local eval_job collect_job
  eval_job=$(sbatch --parsable --job-name="gifteval_${MODEL}_eval" \
    --array="0-${last_task}%8" \
    --dependency="afterok:${array_job}" \
    scripts/eval_pretraining.sbatch)
  collect_job=$(sbatch --parsable --job-name="gifteval_${MODEL}_eval_collect" \
    --dependency="afterok:${eval_job}" \
    scripts/collect_eval.sbatch)
  echo "eval array job ${eval_job} (afterok ${array_job})"
  echo "eval collect job ${collect_job} (afterok ${eval_job})"
  local report_dir
  report_dir="$(output_path analysis "tsfm_eval/${MODEL}" "${JOBTAG}")"
  echo "reports will land at ${report_dir}/eval_report_{main,by_frequency}.md"
}

if [ "${MODEL}" = "chronos2" ] || [ "${MODEL}" = "moirai2" ]; then
  export SEEDS_CSV=${SEEDS_CSV:-0,1,2,3}

  if [ ! -f "${INDEX}" ]; then
    echo "ERROR: window index not found at ${INDEX}" >&2
    exit 1
  fi

  IFS=, read -r -a SEEDS <<< "${SEEDS_CSV}"
  tasks=$((${#SEEDS[@]} * RUNS_PER_SEED))
  last_task=$((tasks - 1))

  ARRAY_JOB=$(sbatch --parsable --job-name="gifteval_${MODEL}" \
    --array="0-${last_task}%8" \
    "${TRAIN_DEP_ARGS[@]}" \
    scripts/run_pretraining.sbatch)

  echo "array job ${ARRAY_JOB}"
  echo "model ${MODEL}"
  echo "tasks ${tasks}"
  echo "seeds ${SEEDS_CSV}"
  echo "maximum concurrent GPU jobs 8"

  export JOBTAG="gifteval_${MODEL}_${ARRAY_JOB}"
  submit_eval_chain "${ARRAY_JOB}" "${last_task}"
  exit 0
fi

export OUTPUT="${INDEX}"

DEPENDENCY_ARGS=()
if [ ! -f "${INDEX}" ]; then
  echo "index not found at ${INDEX}, submitting build job first"
  BUILD_JOB=$(sbatch --parsable scripts/build_gifteval_window_index.sbatch)
  echo "  build job: ${BUILD_JOB}"
  DEPENDENCY_ARGS=(--dependency="afterok:${BUILD_JOB}")
fi

ARRAY_JOB=$(sbatch --parsable --job-name="gifteval_${MODEL}" \
  --array="0-$((RUNS_PER_SEED - 1))" \
  "${DEPENDENCY_ARGS[@]}" scripts/run_pretraining.sbatch)
echo "array job: ${ARRAY_JOB} (${RUNS_PER_SEED} tasks, model=${MODEL}, steps=${STEPS}, batch_size=${BATCH_SIZE}, checkpoint_every=${CHECKPOINT_EVERY})"

export JOBTAG="gifteval_${MODEL}_${ARRAY_JOB}"
AGG_JOB=$(sbatch --parsable --job-name="gifteval_${MODEL}_aggregate" \
  --dependency="afterok:${ARRAY_JOB}" \
  scripts/aggregate_pretraining.sbatch)
echo "aggregate job: ${AGG_JOB} (depends on ${ARRAY_JOB} completing ok)"
echo "outputs will land under $(output_path experiments "tsfm_pretraining/${MODEL}" "${JOBTAG}")"
submit_eval_chain "${ARRAY_JOB}" "$((RUNS_PER_SEED - 1))"

#!/usr/bin/env bash
# Extends the short-series rerun from 60k to 80k updates, continuing from each
# run's checkpoint_step60000.pt. Everything else matches
# scripts/short_series_tsfm_experiment_submit.sh, which produced those
# checkpoints, so the eval chain writes the same reports at the same grains.
#
# The resumed runs land in a new run root (src.training.tsfm refuses to write
# back over the source), and each carries the full 0-to-80k history, so the
# extended runs are self-contained rather than a diff against the 60k ones.
#
# Pass the source array job ids as arguments. MOIRAI_DEP lets the Moirai
# resume queue behind a Moirai array that has not finished yet.
#
# Usage:
#   scripts/short_series_tsfm_experiment_resume.sh 32797 32807
#   MOIRAI_DEP=afterok:32807 scripts/short_series_tsfm_experiment_resume.sh 32797 32807

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/output_paths.sh

# Pinned once and exported so the source run root and the resumed run root
# agree even if submission straddles midnight.
export RUN_DATE=${RUN_DATE:-$(date +%F)}

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <chronos2_array_job> <moirai2_array_job>" >&2
  exit 1
fi

CHRONOS2_JOB=$1
MOIRAI2_JOB=$2
RESUME_STEP=60000
STEPS=80000
CORPUS_ROOT=/zfsauton/scratch/istepka/lts/data/giftevalpretrain_trainsplit
INDEX=outputs/gifteval_window_index/context512_pred128_min64_8_heldout.parquet

submit_one() {
  local model=$1 source_job=$2 dependency=$3
  local resume_root
  resume_root="$(output_path experiments "tsfm_pretraining/${model}" \
    "gifteval_${model}_${source_job}")"
  if [ ! -d "${resume_root}" ]; then
    echo "ERROR: no run root at ${resume_root}" >&2
    exit 1
  fi
  MODEL="${model}" NATURAL_ONLY=1 SEEDS_CSV=0,1,2,3 STEPS="${STEPS}" \
    INDEX="${INDEX}" CORPUS_ROOT="${CORPUS_ROOT}" \
    MIN_CONTEXT_LENGTH=64 MIN_PREDICTION_LENGTH=8 \
    RESUME_ROOT="${resume_root}" RESUME_STEP="${RESUME_STEP}" \
    DEPENDENCY="${dependency}" \
    scripts/submit_pretraining.sh
  echo
}

submit_one chronos2 "${CHRONOS2_JOB}" ""
submit_one moirai2 "${MOIRAI2_JOB}" "${MOIRAI_DEP:-}"

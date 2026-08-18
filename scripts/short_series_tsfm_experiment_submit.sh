#!/usr/bin/env bash
# Submits the short-series rerun of the main-paper TSFM experiment: Chronos-2
# and Moirai-2.0, natural scale only, four seeds, 60k updates at batch size
# 512, on the variable-geometry index over the train-split corpus root.
# Each model's array is followed by the held-out evaluation chain.
#
# These checkpoints are NOT zero-shot on M1, M3, M4, Tourism, or Favorita.
# See notes/08-variable-geometry-implementation.md. The zero-shot table comes
# from jobs 32459 and 32462, which trained on the clean corpus.
#
# Usage:
#   scripts/short_series_tsfm_experiment_submit.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CORPUS_ROOT=/zfsauton/scratch/istepka/lts/data/giftevalpretrain_trainsplit
INDEX=outputs/gifteval_window_index/context512_pred128_min64_8_heldout.parquet

if [ ! -f "${INDEX}" ]; then
  echo "ERROR: variable-geometry index not found at ${INDEX}" >&2
  echo "Build it first:" >&2
  echo "  MIN_CONTEXT_LENGTH=64 MIN_PREDICTION_LENGTH=8 \\" >&2
  echo "    CORPUS_ROOT=${CORPUS_ROOT} \\" >&2
  echo "    sbatch scripts/build_gifteval_window_index.sbatch" >&2
  exit 1
fi

for model in chronos2 moirai2; do
  MODEL="${model}" NATURAL_ONLY=1 SEEDS_CSV=0,1,2,3 STEPS=60000 \
    INDEX="${INDEX}" CORPUS_ROOT="${CORPUS_ROOT}" \
    MIN_CONTEXT_LENGTH=64 MIN_PREDICTION_LENGTH=8 \
    scripts/submit_pretraining.sh
  echo
done

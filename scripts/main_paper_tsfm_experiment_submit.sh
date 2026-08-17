#!/usr/bin/env bash
# Submits the main-paper TSFM experiment: Chronos-2 and Moirai-2.0, natural
# scale only, four seeds, 60k updates at batch size 512. Each model's array is
# followed by the held-out evaluation chain, which ends in the SIT-against-
# RevIN reports.
#
# Usage:
#   scripts/main_paper_tsfm_experiment_submit.sh

set -euo pipefail
cd "$(dirname "$0")/.."

for model in chronos2 moirai2; do
  MODEL="${model}" NATURAL_ONLY=1 SEEDS_CSV=0,1,2,3 STEPS=60000 \
    scripts/submit_pretraining.sh
  echo
done

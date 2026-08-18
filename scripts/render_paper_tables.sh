#!/usr/bin/env bash
# Regenerates the held-out benchmark tables in the extended draft.
#
# The job ids below are the provenance of every number in those tables, so
# they live here rather than in someone's shell history. Change them when the
# tables should follow a newer run, not when a path happens to move.
#
#   zero-shot  32459 / 32462  Chronos-2 and Moirai-2.0, 60k updates, clean
#                             corpus, so all six benchmarks are held out
#   in-domain  32797 / 32807  the same at 60k over the variable-geometry index
#                             with the held-out suites' train regions added
#   baselines  2026-08-17     seasonal naive, ETS, and ARIMA, fitted per series
#
# Usage:
#   scripts/render_paper_tables.sh

set -euo pipefail
cd "$(dirname "$0")/.."

ZS_DATE=2026-08-17
ID_DATE=2026-08-18
BASELINE_DATE=2026-08-17
OUTPUT_DIR=${OUTPUT_DIR:-overleaf/extended_draft/tables}

eval_dir() {
  local date=$1 model=$2 job=$3
  echo "outputs/${date}/analysis/tsfm_eval/${model}/gifteval_${model}_${job}"
}

PYTHONPATH=. uv run python -m src.scripts.render_paper_tables \
  --zero-shot \
    "chronos2=$(eval_dir "${ZS_DATE}" chronos2 32459)" \
    "moirai2=$(eval_dir "${ZS_DATE}" moirai2 32462)" \
  --in-domain \
    "chronos2=$(eval_dir "${ID_DATE}" chronos2 32797)" \
    "moirai2=$(eval_dir "${ID_DATE}" moirai2 32807)" \
  --baselines "outputs/${BASELINE_DATE}/analysis/tsfm_eval/baselines/${BASELINE_DATE}" \
  --output-dir "${OUTPUT_DIR}"

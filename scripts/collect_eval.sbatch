#!/usr/bin/env bash
#SBATCH --job-name=gifteval_eval_collect
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# Merges the per-run tables written by scripts/eval_pretraining.sbatch into
# one comparison of the two loss spaces. Submit with
# --dependency=afterok:<eval_array_job_id> and the same MODEL and JOBTAG,
# which scripts/submit_pretraining.sh does automatically.

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs
source scripts/output_paths.sh

RUN_DATE=${RUN_DATE:-$(date +%F)}

if [ -z "${MODEL:-}" ] || [ -z "${JOBTAG:-}" ]; then
  echo "ERROR: MODEL and JOBTAG must both be set" >&2
  exit 1
fi

EVAL_ROOT="$(output_path analysis "tsfm_eval/${MODEL}" "${JOBTAG}")"

uv run python -m src.scripts.collect_tsfm_eval --eval-root "${EVAL_ROOT}"

echo "report at ${EVAL_ROOT}/eval_report.md"

#!/usr/bin/env bash
#SBATCH --job-name=baseline_eval
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out

# Scores the classical reference baselines over the six held-out suites, one
# array task per baseline. CPU only, so it never competes with training or
# with the model eval for GPUs.
#
# These are reference columns, not zero-shot competitors: each estimator is
# fitted on the series it forecasts. They share the harness's metric code, so
# their numbers sit in the same tables as the checkpoints'.
#
#   sbatch scripts/run_baseline_eval.sbatch

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs
source scripts/output_paths.sh

RUN_DATE=${RUN_DATE:-$(date +%F)}
BASELINES=(seasonal_naive ets arima)
NAME=${BASELINES[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="$(output_path analysis "tsfm_eval/baselines" "${RUN_DATE}")/${NAME}"

echo "baseline ${NAME} -> ${OUT_DIR}"

PYTHONPATH=. uv run python -m src.scripts.run_baseline_eval \
  --baseline "${NAME}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --output-dir "${OUT_DIR}"

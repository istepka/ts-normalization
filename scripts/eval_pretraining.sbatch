#!/usr/bin/env bash
#SBATCH --job-name=gifteval_eval
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out

# Scores each run of a pretraining array against the six held-out evaluation
# suites. Meant to be submitted with --dependency=afterok:<array_job_id> and
# the same MODEL and JOBTAG as that array, which
# scripts/submit_pretraining.sh does automatically.
#
# One array task per run directory. Run directories are discovered by globbing
# RUN_ROOT rather than reconstructed from the RUNS table, so this stays correct
# under NATURAL_ONLY and any seed list. Submit with array bounds matching the
# run count.
#
# The model skeleton comes from each run's own resolved_config.yaml via
# run_dir=, never from conf/eval.yaml's defaults, because a checkpoint only
# loads into the architecture that produced it.

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs
source scripts/output_paths.sh

RUN_DATE=${RUN_DATE:-$(date +%F)}
STEPS=${STEPS:-30000}

case "${MODEL:-}" in
  timesfm | chronos2 | moirai2) ;;
  moment)
    echo "ERROR: MOMENT reconstructs rather than forecasts and has no" >&2
    echo "forecast head, so the eval harness does not score it." >&2
    exit 1
    ;;
  *)
    echo "ERROR: MODEL must be timesfm, chronos2, or moirai2" >&2
    exit 1
    ;;
esac

if [ -z "${JOBTAG:-}" ]; then
  echo "ERROR: JOBTAG must be set, matching the pretraining array's JOBTAG" >&2
  exit 1
fi

RUN_ROOT="$(output_path experiments "tsfm_pretraining/${MODEL}" "${JOBTAG}")"
mapfile -t RUN_DIRS < <(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
  echo "ERROR: no run directories under ${RUN_ROOT}" >&2
  exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#RUN_DIRS[@]}" ]; then
  echo "ERROR: task ${SLURM_ARRAY_TASK_ID} but only ${#RUN_DIRS[@]} runs" >&2
  exit 1
fi

RUN_DIR="${RUN_DIRS[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT="${RUN_DIR}/checkpoint_step${STEPS}.pt"
OUT_DIR="$(output_path analysis "tsfm_eval/${MODEL}" "${JOBTAG}")/$(basename "${RUN_DIR}")"

if [ ! -f "${CHECKPOINT}" ]; then
  echo "ERROR: no checkpoint at ${CHECKPOINT}" >&2
  exit 1
fi

echo "task ${SLURM_ARRAY_TASK_ID}: ${RUN_DIR} -> ${OUT_DIR}"

PYTHONPATH=. uv run python -m src.scripts.run_tsfm_eval \
  device=cuda \
  "run_dir=${RUN_DIR}" \
  "checkpoint=${CHECKPOINT}" \
  "output_dir=${OUT_DIR}" \
  "hydra.run.dir=${OUT_DIR}"

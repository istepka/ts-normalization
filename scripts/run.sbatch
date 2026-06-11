#!/usr/bin/env bash
#SBATCH --job-name=loss_space
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=30:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

# Duplicate-submission guard: if another job with the same name is queued/running,
# only the one with the lowest JobID proceeds; the rest exit immediately. This is
# race-proof even when the same job is submitted twice within seconds.
others=$(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t PENDING,RUNNING -o "%i" \
  | grep -v "^${SLURM_JOB_ID}$" || true)
for jid in $others; do
  if [ "${jid%%_*}" -lt "$SLURM_JOB_ID" ]; then
    echo "Duplicate of older job ${jid} (this=${SLURM_JOB_ID}); exiting without running."
    exit 0
  fi
done

# Any extra args are passed through as Hydra overrides, e.g.:
#   sbatch scripts/run.sbatch train.steps=8000 train.lr=3e-4
uv run python main.py device=cuda "$@"

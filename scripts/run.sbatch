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

# Job-ID-prefixed output dir so concurrent variants never collide; the paper dir
# (`${RUN_DIR}_paper`) inherits the same prefix automatically.
RUN_DIR="outputs/${SLURM_JOB_ID}_loss_space_toy"

# Any extra args are passed through as Hydra overrides, e.g. a longer variant:
#   sbatch --job-name=loss_space_100k scripts/run.sbatch train.steps=100000
# (give a distinct --job-name when running variants concurrently, so the guard
# above does not treat them as duplicates of each other).
uv run python main.py device=cuda output_dir="$RUN_DIR" "$@"
echo "outputs: $RUN_DIR  (paper: ${RUN_DIR}_paper)"

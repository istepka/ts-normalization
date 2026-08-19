#!/bin/bash
#SBATCH --account=users
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -euo pipefail

repo=${SLURM_SUBMIT_DIR:?submit from the repository root}
cd "${repo}"
test -f src/training/evaluate_supervised.py
uv run python -m src.training.evaluate_supervised "$@"

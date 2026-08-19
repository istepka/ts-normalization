#!/bin/bash
#SBATCH --account=users
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -euo pipefail

cd /zfsauton/scratch/istepka/tmp/worktrees/m-series-supervised
uv run python -m src.training.evaluate_supervised "$@"

#!/usr/bin/env bash
#SBATCH --job-name=install_forecast_figs
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

uv run python scripts/reproducibility/synthetic_loss_space/install_paper_figures.py \
  --sgd-dir=outputs/23555_loss_space_toy \
  --adam-dir=outputs/23556_loss_space_toy \
  --figure-root=overleaf/figures/loss_space/synthetic_loss_space

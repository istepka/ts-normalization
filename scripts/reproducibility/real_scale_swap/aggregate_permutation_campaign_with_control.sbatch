#!/usr/bin/env bash
#SBATCH --job-name=scale_perm_ctrl
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

uv run python -m scripts.reproducibility.real_scale_swap.aggregate_permutation_campaign \
  --campaign-root=outputs/scale_swap_permutations \
  --normalized-campaign-root=outputs/scale_swap_permutations_normalized \
  --output-dir=outputs/scale_swap_permutation_analysis

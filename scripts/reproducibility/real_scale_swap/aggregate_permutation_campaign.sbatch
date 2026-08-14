#!/usr/bin/env bash
#SBATCH --job-name=scale_perm_agg
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
source scripts/output_paths.sh

# Defaults point at the completed campaign; override for a new one.
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-outputs/2026-07-29/experiments/legacy_runs/scale_swap_permutations}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$(output_path analysis real_scale_swap scale_swap_permutation_analysis)}"

uv run python -m src.plotting.scripts.reproducibility.real_scale_swap.aggregate_permutation_campaign \
  --campaign-root="${CAMPAIGN_ROOT}" \
  --output-dir="${ANALYSIS_DIR}"

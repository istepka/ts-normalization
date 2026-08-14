#!/usr/bin/env bash
#SBATCH --job-name=scale_swap_lr_adj
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs
source scripts/output_paths.sh

RUN_ROOT="$(output_path experiments real_scale_swap "job-${SLURM_JOB_ID}/lr_adjusted")"

labels=(a b)
assignments=(
  '[1.0,1.0,1.0,1.0,10.0,10.0,10.0,10.0]'
  '[10.0,10.0,10.0,10.0,1.0,1.0,1.0,1.0]'
)

run_dirs=()
for index in "${!labels[@]}"; do
  label="${labels[index]}"
  assignment="${assignments[index]}"
  run_dir="${RUN_ROOT}/${label}"
  echo "running LR-adjusted assignment ${label}: ${assignment}"
  uv run python main.py --config-name=scale_swap \
    device=cuda \
    setups='[original_lr_adjusted]' \
    "output_dir=${run_dir}" \
    "hydra.run.dir=${run_dir}" \
    "wandb.experiment=real_scale_swap_lr_adjusted_${label}" \
    "data.scale_assignment=${assignment}"
  run_dirs+=("${run_dir}")
done

aggregate_dir="$(output_path analysis real_scale_swap "job-${SLURM_JOB_ID}/scale_swap_lr_adjusted_comparison")"
uv run python -m src.plotting.scripts.reproducibility.real_scale_swap.aggregate_crossover \
  --assignment-a=outputs/2026-07-17/experiments/legacy_runs/18948_scale_swap_a \
  --assignment-b=outputs/2026-07-17/experiments/legacy_runs/18948_scale_swap_b \
  --lr-adjusted-a="${run_dirs[0]}" \
  --lr-adjusted-b="${run_dirs[1]}" \
  --output-dir="${aggregate_dir}"

echo "LR-adjusted assignment A output: ${run_dirs[0]}"
echo "LR-adjusted assignment B output: ${run_dirs[1]}"
echo "comparison output: ${aggregate_dir}"

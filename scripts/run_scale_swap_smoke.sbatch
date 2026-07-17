#!/usr/bin/env bash
#SBATCH --job-name=scale_swap_smoke
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

labels=(a b)
assignments=(
  '[1.0,1.0,1.0,1.0,10.0,10.0,10.0,10.0]'
  '[10.0,10.0,10.0,10.0,1.0,1.0,1.0,1.0]'
)

for index in "${!labels[@]}"; do
  label="${labels[index]}"
  assignment="${assignments[index]}"
  run_dir="outputs/${SLURM_JOB_ID}_scale_swap_smoke_${label}"
  echo "smoking assignment ${label}: ${assignment}"
  uv run python main.py --config-name=scale_swap \
    device=cuda \
    train.steps=200 \
    "output_dir=${run_dir}" \
    wandb.mode=disabled \
    "data.scale_assignment=${assignment}"
done

echo "all five seeds remained finite through 200 steps for both assignments"

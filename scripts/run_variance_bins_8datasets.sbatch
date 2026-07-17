#!/usr/bin/env bash
#SBATCH --job-name=varbins8_all
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

data_root=/zfsauton2/home/istepka/sstc/experiments/h1/data
datasets=(
  electricity
  traffic
  solar_10_minutes
  taxi_30min
  wind_farms_without_missing
  pedestrian_counts
  kdd_cup_2018_without_missing
  fred_md
)

# Training-only p99 context std is mapped to 0.6475868966 for every dataset.
# This matches Electricity at real_value_scale=0.0001 and preserves each
# dataset's internal low/mid/high variance ratios.
scales=(
  0.0001
  8.59743634
  0.06950270192
  0.0412493237
  0.01258875565
  0.0004366240309
  0.003212349661
  0.00001744783949
)

aggregate_args=()
for index in "${!datasets[@]}"; do
  dataset="${datasets[index]}"
  scale="${scales[index]}"
  run_dir="outputs/${SLURM_JOB_ID}_${dataset}_loss_space_toy"
  echo "running ${dataset} real_value_scale=${scale} output=${run_dir}"
  uv run python main.py \
    device=cuda \
    "output_dir=${run_dir}" \
    "wandb.experiment=real_variance_bins_8datasets_${dataset}_splitfix" \
    data.kind=real_variance_binned \
    "data.real_shape_path=${data_root}/128_${dataset}.npz" \
    data.real_shape_key=data \
    "data.real_value_scale=${scale}"
  aggregate_args+=("${dataset}=${run_dir}")
done

aggregate_dir="outputs/${SLURM_JOB_ID}_variance_bins_8datasets"
uv run python scripts/aggregate_variance_bins.py \
  "${aggregate_args[@]}" \
  --output-dir="${aggregate_dir}"

echo "per-dataset outputs: outputs/${SLURM_JOB_ID}_*_loss_space_toy"
echo "aggregate output: ${aggregate_dir}"

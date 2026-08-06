#!/usr/bin/env bash
#SBATCH --job-name=varbins8_recover
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

data_root=/zfsauton2/home/istepka/sstc/experiments/h1/data
datasets=(
  kdd_cup_2018_without_missing
  fred_md
)
scales=(
  0.003212349661
  0.00001744783949
)

aggregate_args=(
  electricity=outputs/18857_electricity_loss_space_toy
  traffic=outputs/18857_traffic_loss_space_toy
  solar_10_minutes=outputs/18857_solar_10_minutes_loss_space_toy
  taxi_30min=outputs/18857_taxi_30min_loss_space_toy
  wind_farms_without_missing=outputs/18857_wind_farms_without_missing_loss_space_toy
  pedestrian_counts=outputs/18857_pedestrian_counts_loss_space_toy
)

for index in "${!datasets[@]}"; do
  dataset="${datasets[index]}"
  scale="${scales[index]}"
  run_dir="outputs/${SLURM_JOB_ID}_${dataset}_loss_space_toy"
  echo "running ${dataset} real_value_scale=${scale} output=${run_dir}"
  uv run python main.py \
    device=cuda \
    "output_dir=${run_dir}" \
    "wandb.experiment=varbins8_${dataset}_splitfix" \
    data.kind=real_variance_binned \
    "data.real_shape_path=${data_root}/128_${dataset}.npz" \
    data.real_shape_key=data \
    "data.real_value_scale=${scale}"
  aggregate_args+=("${dataset}=${run_dir}")
done

aggregate_dir="outputs/${SLURM_JOB_ID}_variance_bins_8datasets"
uv run python scripts/reproducibility/real_variance_bins/aggregate.py \
  "${aggregate_args[@]}" \
  --output-dir="${aggregate_dir}"

echo "recovered outputs: outputs/${SLURM_JOB_ID}_*_loss_space_toy"
echo "aggregate output: ${aggregate_dir}"

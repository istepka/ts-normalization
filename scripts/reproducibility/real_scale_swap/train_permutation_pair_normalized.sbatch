#!/usr/bin/env bash
#SBATCH --job-name=scale_perm_norm
#SBATCH --partition=legacy
#SBATCH --qos=qos_legacy
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs outputs/scale_swap_permutations_normalized

pair_index="$1"
if ((pair_index < 2 || pair_index > 15)); then
  echo "pair index must be between 2 and 15; pair 1 is completed job 18948" >&2
  exit 2
fi
printf -v pair_tag '%02d' "${pair_index}"

for variant in a b; do
  assignment="$(uv run python -m src.scripts.permutation_schedule \
    --pair-index="${pair_index}" --arm="${variant}")"
  run_dir="outputs/scale_swap_permutations_normalized/pair_${pair_tag}_${variant}"
  if [[ -e "${run_dir}" ]]; then
    echo "refusing to overwrite existing output: ${run_dir}" >&2
    exit 2
  fi
  echo "running normalized pair ${pair_tag} variant ${variant}: ${assignment}"
  uv run python main.py --config-name=scale_swap \
    device=cuda \
    setups='[normalized]' \
    "output_dir=${run_dir}" \
    "wandb.experiment=scale_perm_norm_p${pair_tag}_${variant}" \
    "data.scale_assignment=${assignment}"
done

echo "completed normalized complementary pair ${pair_tag}"

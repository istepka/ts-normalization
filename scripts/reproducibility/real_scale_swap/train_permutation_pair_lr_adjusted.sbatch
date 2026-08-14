#!/usr/bin/env bash
#SBATCH --job-name=scale_perm
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
mkdir -p logs
source scripts/output_paths.sh

# One campaign spans 14 submissions, so every pair must land under the same
# root. Set CAMPAIGN_ROOT (or OUTPUT_DATE) to keep them together across days;
# the default only works if the whole campaign is submitted on one day.
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$(output_path experiments real_scale_swap scale_swap_permutations)}"
mkdir -p "${CAMPAIGN_ROOT}"

pair_index="$1"
if ((pair_index < 2 || pair_index > 15)); then
  echo "pair index must be between 2 and 15; pair 1 is completed job 18985" >&2
  exit 2
fi
printf -v pair_tag '%02d' "${pair_index}"

for arm in a b; do
  assignment="$(uv run python -m src.scripts.permutation_schedule \
    --pair-index="${pair_index}" --arm="${arm}")"
  run_dir="${CAMPAIGN_ROOT}/pair_${pair_tag}_${arm}"
  if [[ -e "${run_dir}" ]]; then
    echo "refusing to overwrite existing output: ${run_dir}" >&2
    exit 2
  fi
  echo "running pair ${pair_tag} arm ${arm}: ${assignment}"
  uv run python main.py --config-name=scale_swap \
    device=cuda \
    setups='[original_lr_adjusted]' \
    "output_dir=${run_dir}" \
    "hydra.run.dir=${run_dir}" \
    "wandb.experiment=scale_perm_p${pair_tag}_${arm}" \
    "data.scale_assignment=${assignment}"
done

echo "completed complementary pair ${pair_tag}"

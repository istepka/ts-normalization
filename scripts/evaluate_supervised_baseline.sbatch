#!/usr/bin/env bash
#SBATCH --job-name=mseries_baseline
#SBATCH --partition=cpu
#SBATCH --account=users
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-19%20
#SBATCH --output=/zfsauton/scratch/istepka/ts-normalization/logs/supervised_baselines/%A_%a.out
#SBATCH --error=/zfsauton/scratch/istepka/ts-normalization/logs/supervised_baselines/%A_%a.err

set -euo pipefail

repo=${SLURM_SUBMIT_DIR:?submit from the repository root}
output=/zfsauton/scratch/istepka/ts-normalization/outputs/supervised_baselines
cd "${repo}"
test -f src/training/evaluate_supervised_baseline.py
task=${SLURM_ARRAY_TASK_ID}
tasks=20
cursor=0

for baseline in ets arima; do
    for specification in \
        "m1 Y 1" "m1 Q 1" "m1 M 1" \
        "m3 Y 1" "m3 Q 1" "m3 M 1" \
        "m4 Y 8" "m4 Q 23" "m4 M 33" \
        "m4 W 1" "m4 D 5" "m4 H 1" \
        "tourism Y 1" "tourism Q 1" "tourism M 1"; do
        read -r suite frequency shards <<< "${specification}"
        for ((shard = 0; shard < shards; shard++)); do
            if ((cursor % tasks == task)); then
                uv run python -m src.training.evaluate_supervised_baseline \
                    --baseline "${baseline}" \
                    --suite "${suite}" \
                    --frequency "${frequency}" \
                    --shard-index "${shard}" \
                    --num-shards "${shards}" \
                    --workers "${SLURM_CPUS_PER_TASK}" \
                    --output-dir "${output}"
            fi
            ((cursor += 1))
        done
    done
done

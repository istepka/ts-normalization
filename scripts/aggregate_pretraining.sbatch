#!/usr/bin/env bash
#SBATCH --job-name=gifteval_aggregate
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# Aggregates the 6 runs produced by run_pretraining.sbatch's job array for
# MODEL=moment or MODEL=timesfm. Meant to be submitted with
# --dependency=afterok:<array_job_id> and JOBTAG matching that array job's
# JOBTAG (scripts/submit_pretraining.sh does both automatically).

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

if [ -z "${JOBTAG:-}" ]; then
  echo "ERROR: JOBTAG must be set (matching the run_pretraining.sbatch array job's JOBTAG)." >&2
  exit 1
fi

case "${MODEL:-}" in
  moment)
    COND_A=moment_normalized
    COND_B=moment_original
    ;;
  timesfm)
    COND_A=timesfm_native_original
    COND_B=timesfm_normalized
    ;;
  *)
    echo "ERROR: MODEL must be set to moment or timesfm" >&2
    exit 1
    ;;
esac

uv run python -m scripts.aggregate_tsfm_loss_space \
  --run "${COND_A}=outputs/${JOBTAG}_natural_${COND_A}" \
  --run "${COND_B}=outputs/${JOBTAG}_natural_${COND_B}" \
  --output-dir "outputs/${JOBTAG}_natural_aggregate"

for condition in "${COND_A}" "${COND_B}"; do
  uv run python -m scripts.aggregate_tsfm_loss_space \
    --run "${condition}_A=outputs/${JOBTAG}_scale_${condition}_A" \
    --run "${condition}_B=outputs/${JOBTAG}_scale_${condition}_B" \
    --scale-pair "${condition}_A=${condition}_B" \
    --output-dir "outputs/${JOBTAG}_scale_${condition}_aggregate"
done

echo "natural-mixture aggregate: outputs/${JOBTAG}_natural_aggregate"
echo "controlled-scale aggregates: outputs/${JOBTAG}_scale_{${COND_A},${COND_B}}_aggregate"

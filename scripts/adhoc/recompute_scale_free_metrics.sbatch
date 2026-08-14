#!/usr/bin/env bash
#SBATCH --job-name=tsfm_scale_free_metrics
#SBATCH --partition=project
#SBATCH --qos=qos_radio_freq
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# Rescores every saved checkpoint of the 12 GiftEvalPretrain runs on
# scale-free metrics (nMSE, MASE), replacing the original per-variant
# training-space MSE that made the normalized and original variants
# incomparable. See src/tsfm_pretraining/scripts/recompute_tsfm_scale_free_metrics.py.

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

MOMENT_TAG=gifteval_moment_26371
TIMESFM_TAG=gifteval_timesfm_fixed

uv run python -m src.tsfm_pretraining.scripts.recompute_tsfm_scale_free_metrics \
  --run "moment_normalized_natural=outputs/${MOMENT_TAG}_natural_moment_normalized" \
  --run "moment_original_natural=outputs/${MOMENT_TAG}_natural_moment_original" \
  --run "moment_normalized_A=outputs/${MOMENT_TAG}_scale_moment_normalized_A" \
  --run "moment_normalized_B=outputs/${MOMENT_TAG}_scale_moment_normalized_B" \
  --run "moment_original_A=outputs/${MOMENT_TAG}_scale_moment_original_A" \
  --run "moment_original_B=outputs/${MOMENT_TAG}_scale_moment_original_B" \
  --run "timesfm_normalized_natural=outputs/${TIMESFM_TAG}_natural_timesfm_normalized" \
  --run "timesfm_original_natural=outputs/${TIMESFM_TAG}_natural_timesfm_native_original" \
  --run "timesfm_normalized_A=outputs/${TIMESFM_TAG}_scale_timesfm_normalized_A" \
  --run "timesfm_normalized_B=outputs/${TIMESFM_TAG}_scale_timesfm_normalized_B" \
  --run "timesfm_original_A=outputs/${TIMESFM_TAG}_scale_timesfm_native_original_A" \
  --run "timesfm_original_B=outputs/${TIMESFM_TAG}_scale_timesfm_native_original_B" \
  --output-dir outputs/tsfm_scale_free_metrics_v2

echo "wrote outputs/tsfm_scale_free_metrics_v2"

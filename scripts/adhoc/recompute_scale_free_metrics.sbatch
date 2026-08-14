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
# incomparable. See src/scripts/recompute_tsfm_scale_free_metrics.py.

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs

# These runs predate the dated output layout and now live under legacy_runs.
LEGACY_ROOT=outputs/2026-08-10/experiments/legacy_runs
MOMENT_TAG=gifteval_moment_26371
TIMESFM_TAG=gifteval_timesfm_fixed
METRICS_DIR="${METRICS_DIR:-$(output_path analysis tsfm_pretraining tsfm_scale_free_metrics_v2)}"

uv run python -m src.scripts.recompute_tsfm_scale_free_metrics \
  --run "moment_normalized_natural=${LEGACY_ROOT}/${MOMENT_TAG}_natural_moment_normalized" \
  --run "moment_original_natural=${LEGACY_ROOT}/${MOMENT_TAG}_natural_moment_original" \
  --run "moment_normalized_A=${LEGACY_ROOT}/${MOMENT_TAG}_scale_moment_normalized_A" \
  --run "moment_normalized_B=${LEGACY_ROOT}/${MOMENT_TAG}_scale_moment_normalized_B" \
  --run "moment_original_A=${LEGACY_ROOT}/${MOMENT_TAG}_scale_moment_original_A" \
  --run "moment_original_B=${LEGACY_ROOT}/${MOMENT_TAG}_scale_moment_original_B" \
  --run "timesfm_normalized_natural=${LEGACY_ROOT}/${TIMESFM_TAG}_natural_timesfm_normalized" \
  --run "timesfm_original_natural=${LEGACY_ROOT}/${TIMESFM_TAG}_natural_timesfm_native_original" \
  --run "timesfm_normalized_A=${LEGACY_ROOT}/${TIMESFM_TAG}_scale_timesfm_normalized_A" \
  --run "timesfm_normalized_B=${LEGACY_ROOT}/${TIMESFM_TAG}_scale_timesfm_normalized_B" \
  --run "timesfm_original_A=${LEGACY_ROOT}/${TIMESFM_TAG}_scale_timesfm_native_original_A" \
  --run "timesfm_original_B=${LEGACY_ROOT}/${TIMESFM_TAG}_scale_timesfm_native_original_B" \
  --output-dir "${METRICS_DIR}"

echo "wrote ${METRICS_DIR}"

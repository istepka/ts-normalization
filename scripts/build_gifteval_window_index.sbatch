#!/usr/bin/env bash
#SBATCH --job-name=gifteval_window_index
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# CPU-only prerequisite for run_pretraining.sbatch (MODEL=moment or
# MODEL=timesfm): builds and caches the canonical window index once so
# neither training job burns an expensive GPU allocation on scanning the
# corpus, and so every paired condition trains on identical windows (see the
# "Canonical window index" section of
# notes/05-timesfm-pretraining-loss-space-plan.md).
#
# Scanning is CPU/IO-bound, not GPU-bound; the 24h budget is deliberately
# generous and untuned -- the largest single dataset in the corpus
# (buildings_900k, 1.8M series) alone took multiple minutes to scan on a
# lightly-loaded login-node smoke test, and this job additionally covers
# every other univariate dataset in the corpus (see notes/agentic_logs for
# the smoke-test numbers this estimate is based on). Time this job's actual
# wall clock on first submission and tighten --time for reruns.

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization
mkdir -p logs
source scripts/output_paths.sh

RUN_DATE=${RUN_DATE:-$(date +%F)}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-512}
PREDICTION_LENGTH=${PREDICTION_LENGTH:-128}
STRIDE=${STRIDE:-512}
# The window index is a shared cache, not a run artifact: every paired run must
# read the same file, so it lives at a stable path rather than a dated one.
INDEX_DIR=${INDEX_DIR:-outputs/gifteval_window_index}
# The default carries the _heldout suffix because the default --exclude below
# is non-empty: an index built here is a held-out index, and training raises on
# one that still contains the excluded datasets.
OUTPUT=${OUTPUT:-${INDEX_DIR}/context${CONTEXT_LENGTH}_pred${PREDICTION_LENGTH}_heldout.parquet}
mkdir -p "$(dirname "${OUTPUT}")"

uv run python -m src.scripts.build_gifteval_window_index \
  --corpus-root /zfsauton/scratch/istepka/lts/data/giftevalpretrain_full \
  --output "${OUTPUT}" \
  --context-length "${CONTEXT_LENGTH}" \
  --prediction-length "${PREDICTION_LENGTH}" \
  --stride "${STRIDE}" \
  --exclude "${EXCLUDE:-m1_yearly,m1_quarterly,m1_monthly,monash_m3_yearly,monash_m3_quarterly,monash_m3_monthly,monash_m3_other,tourism_yearly,tourism_quarterly,tourism_monthly,favorita_sales,favorita_transactions}"

echo "window index ready at ${OUTPUT}"
echo "pass window_index.cache_path=${OUTPUT} to run_pretraining.sbatch"

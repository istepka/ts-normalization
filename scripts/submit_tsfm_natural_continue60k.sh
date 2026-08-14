#!/usr/bin/env bash

set -euo pipefail
cd /zfsauton2/home/istepka/ts-normalization

index=outputs/gifteval_window_index/context512_pred128.parquet
[[ -f $index ]]

check_run_dir() {
  local source_dir=$1
  test -f "${source_dir}/checkpoint_step30000.pt"
  test -f "${source_dir}/history.json"
  test -f "${source_dir}/summary.json"
}

for seed in 0 1 2 3; do
  for condition in moment_normalized moment_original; do
    check_run_dir "outputs/2026-08-11/experiments/legacy_runs/gifteval_moment_28827_seed${seed}_${condition}_natural"
  done
  for condition in timesfm_normalized timesfm_native_original; do
    check_run_dir "outputs/2026-08-13/experiments/tsfm_pretraining/timesfm/timesfm_natural_eval250_30192/seed${seed}_${condition}_natural"
  done
  for condition in chronos2_normalized chronos2_original; do
    check_run_dir "outputs/2026-08-12/experiments/tsfm_pretraining/chronos2/gifteval_chronos2_b512_29436/seed${seed}_${condition}_natural"
  done
  for condition in moirai2_normalized moirai2_original; do
    check_run_dir "outputs/2026-08-12/experiments/tsfm_pretraining/moirai2/gifteval_moirai2_b512_29437/seed${seed}_${condition}_natural"
  done
done

if [[ ${1:-} == --check-only ]]; then
  echo "Validated 32 source checkpoints and the window-index cache."
  exit 0
fi

ssh rhea sbatch /zfsauton2/home/istepka/ts-normalization/scripts/adhoc/run_tsfm_natural_continue60k.sbatch

#!/bin/bash

set -euo pipefail

worktree=/zfsauton/scratch/istepka/tmp/worktrees/m-series-supervised
output_root=${1:?usage: submit_supervised_evaluation.sh OUTPUT_ROOT}

for model in nhits nbeats patchtst; do
  for condition in sit revin; do
    for normalization in standard causal; do
      for frequency in Y Q M W D H; do
        case "$frequency" in
          M|H)
            partition=general
            gpu_request=gpu:1
            ;;
          Y|Q|W|D)
            partition=legacy
            gpu_request=gpu:rtx_2080_ti:1
            ;;
        esac
        run_dir="$output_root/$model/$frequency/$condition/$normalization/seed0"
        sbatch \
          --partition="$partition" \
          --gres="$gpu_request" \
          --job-name="eval_${model}_${frequency}_${condition}_${normalization}" \
          "$worktree/scripts/evaluate_supervised.sbatch" \
          --run-dir "$run_dir" \
          --device cuda
      done
    done
  done
done

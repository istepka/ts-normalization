#!/bin/bash

set -euo pipefail

worktree=/zfsauton/scratch/istepka/tmp/worktrees/m-series-supervised
requested_partition=${1:-all}

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
        if [[ "$requested_partition" != all \
          && "$partition" != "$requested_partition" ]]; then
          continue
        fi
        sbatch \
          --partition="$partition" \
          --gres="$gpu_request" \
          --job-name="sup_${model}_${frequency}_${condition}_${normalization}" \
          "$worktree/scripts/train_supervised.sbatch" \
          model="${model}" \
          condition="${condition}" \
          normalization="${normalization}" \
          frequency="${frequency}" \
          seed=0
      done
    done
  done
done

#!/usr/bin/env bash

set -euo pipefail

output_path() {
  if (( $# != 3 )); then
    echo "usage: output_path CATEGORY EXPERIMENT RUN" >&2
    return 2
  fi

  local category="$1"
  local experiment="$2"
  local run="$3"
  local run_date="${OUTPUT_DATE:-${RUN_DATE:-$(date +%F)}}"

  case "${category}" in
    analysis | data | diagnostics | experiments | visualizations) ;;
    *)
      echo "unknown output category: ${category}" >&2
      return 2
      ;;
  esac

  if [[ "${experiment}" == /* || "${experiment}" == *..* ]]; then
    echo "experiment must be a relative path without '..': ${experiment}" >&2
    return 2
  fi
  if [[ "${run}" == /* || "${run}" == *..* ]]; then
    echo "run must be a relative path without '..': ${run}" >&2
    return 2
  fi

  printf 'outputs/%s/%s/%s/%s\n' "${run_date}" "${category}" "${experiment}" "${run}"
}

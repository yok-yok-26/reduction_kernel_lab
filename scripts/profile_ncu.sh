#!/usr/bin/env bash
set -euo pipefail

n="${1:-16777216}"
mode_arg="${2:-all}"
mkdir -p reports/ncu

modes=(library_cub_device_reduce custom_cuda_atomic optimized_todo optimized_tile2 optimized_tile4 optimized_tile8 optimized_2stage optimized_2stage_tile)
if [[ "${mode_arg}" != "all" ]]; then
  modes=("${mode_arg}")
fi

for mode in "${modes[@]}"; do
  ncu \
    -f \
    --set full \
    -o "reports/ncu/reduction_${mode}_single_latest" \
    ./build-release/profile_reduction_once "${mode}" "${n}"
done

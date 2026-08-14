#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DDEBUG_CUDA_SYNC=ON \
  ${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}}

cmake --build build-debug -j

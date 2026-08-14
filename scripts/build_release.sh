#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  ${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}}

cmake --build build-release -j

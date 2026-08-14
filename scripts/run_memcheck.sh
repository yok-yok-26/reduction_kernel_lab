#!/usr/bin/env bash
set -euo pipefail

mkdir -p reports/memcheck
compute-sanitizer --tool memcheck \
  --error-exitcode 1 \
  --log-file reports/memcheck/latest.log \
  ./build-debug/test_reduction_correctness

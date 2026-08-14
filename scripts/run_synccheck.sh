#!/usr/bin/env bash
set -euo pipefail

mkdir -p reports/synccheck
compute-sanitizer --tool synccheck \
  --error-exitcode 1 \
  --log-file reports/synccheck/latest.log \
  ./build-debug/test_reduction_correctness

#!/usr/bin/env bash
set -euo pipefail

mkdir -p reports/racecheck
compute-sanitizer --tool racecheck \
  --error-exitcode 1 \
  --log-file reports/racecheck/latest.log \
  ./build-debug/test_reduction_correctness

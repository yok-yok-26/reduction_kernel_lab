#!/usr/bin/env bash
set -euo pipefail

mkdir -p reports/benchmark
./build-release/bench_reduction "${1:-16777216}" "${2:-100}" | tee reports/benchmark/latest.txt

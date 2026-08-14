#!/usr/bin/env bash
set -euo pipefail

mkdir -p reports/correctness
./build-debug/test_reduction_correctness   > >(tee reports/correctness/latest.txt)   2> >(tee reports/correctness/numeric_notes.log >&2)

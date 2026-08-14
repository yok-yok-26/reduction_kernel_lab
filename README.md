# Reduction Kernel Lab

CUDA reduction kernel learning project for comparing hand-written reduction kernels with a fair generic library baseline.

The project is structured so the CUDA kernel implementations live in `kernels/`, while the surrounding build, reference, correctness, benchmark, sanitizer, and profiling harnesses are kept in normal project directories.

## Contents

- `kernels/`: CUDA reduction kernels and launch interfaces.
- `reference/`: CPU reference implementation.
- `tests/`: correctness tests, benchmark entry points, and single-operation profiler entry point.
- `scripts/`: build, validation, benchmark, sanitizer, profiling, and plot-generation scripts.
- `reports/benchmark/*.csv`: public benchmark summary data.
- `reports/final_overview/images_latest/`: public performance display charts.

Raw profiler reports, sanitizer logs, correctness logs, build directories, and local analysis notes are intentionally excluded from Git.

## Requirements

Validated environment:

- Ubuntu Linux
- NVIDIA GPU: NVIDIA GeForce RTX 5070
- NVIDIA driver: 580.173.02
- CUDA toolkit: `/usr/local/cuda-12.8/bin/nvcc`, CUDA 12.8.61
- CMake 3.18 or newer

The release build used for the current benchmark artifacts was built with:

```bash
CUDA_ARCH=120 ./scripts/build_release.sh
```

## Build

Debug build:

```bash
./scripts/build_debug.sh
```

Release build:

```bash
CUDA_ARCH=120 ./scripts/build_release.sh
```

## Run

Correctness:

```bash
./scripts/run_correctness.sh
```

Benchmark:

```bash
./scripts/run_benchmark.sh
```

The benchmark compares:

- `library_cub_device_reduce`: L1 fair generic library baseline using CUB `DeviceReduce`.
- `custom_cuda_atomic`: custom CUDA atomic comparison, not a fair baseline.
- `optimized_todo`
- `optimized_tile2`
- `optimized_tile4`
- `optimized_tile8`
- `optimized_2stage`
- `optimized_2stage_tile`

## Verify

The normal validation flow is:

```bash
./scripts/build_debug.sh
./scripts/run_correctness.sh
./scripts/run_memcheck.sh
CUDA_ARCH=120 ./scripts/build_release.sh
./scripts/run_benchmark.sh
./scripts/run_racecheck.sh
./scripts/run_synccheck.sh
```

Profiler scripts are available for local CUDA development environments with Nsight Systems and Nsight Compute:

```bash
./scripts/profile_nsys.sh
./scripts/profile_ncu.sh
```

Profiler raw outputs are intentionally not tracked in Git.

## Benchmark Notes

Benchmark timing uses release builds, not debug `-G` builds. Device memory, CUB temporary storage, and user workspaces are allocated outside the timed loop when supported by the implementation. Release benchmark timings are kept separate from Nsight Compute replay timings.

Current public summary data is stored in:

- `reports/benchmark/summary_latest.csv`
- `reports/benchmark/multiscale/reduction_20scale_latest.csv`
- `reports/benchmark/batched/reduction_batched_latest.csv`
- `reports/benchmark/batched/reduction_batched_summary_latest.csv`

Current public display charts are stored in:

- `reports/final_overview/images_latest/reduction_best_user_vs_baseline.svg`
- `reports/final_overview/images_latest/reduction_best_user_ratio.svg`
- `reports/final_overview/images_latest/reduction_full_series_heatmap.svg`
- `reports/final_overview/images_latest/reduction_ncu_metrics_by_algorithm.svg`
- `reports/final_overview/images_latest/reduction_ncu_metrics_by_metric.svg`
- `reports/final_overview/images_latest/reduction_stall_top5_by_algorithm.svg`
- `reports/final_overview/images_latest/reduction_latest.png`
- `reports/final_overview/images_latest/reduction_global_latest.png`

## License

Apache License 2.0. See `LICENSE`.

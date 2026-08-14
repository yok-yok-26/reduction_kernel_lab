#include "kernels/cuda_check.cuh"
#include "kernels/reduction_kernel.cuh"
#include "reference/cpu_reduce.h"
#include "tests/reduction_library_baselines.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

std::vector<float> MakeInput(int n) {
    std::vector<float> x(n);
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        x[i] = dist(rng);
    }
    return x;
}

double Percentile(std::vector<float> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    double pos = (values.size() - 1) * p;
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return values[lo];
    double frac = pos - lo;
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

std::size_t MaxBenchmarkWorkspaceElements(int n) {
    std::size_t max_elements = 0;
    max_elements = std::max(max_elements, GetReduceSumWorkspaceElementCount(n, ReductionKernelMode::BaselineAtomic));
    max_elements = std::max(max_elements, GetReduceSumWorkspaceElementCount(n, ReductionKernelMode::OptimizedTodo));
    max_elements = std::max(max_elements, GetReduceSumWorkspaceElementCount(n, ReductionKernelMode::OptimizedTile2));
    max_elements = std::max(max_elements, GetReduceSumWorkspaceElementCount(n, ReductionKernelMode::OptimizedTile4));
    max_elements = std::max(max_elements, GetReduceSumWorkspaceElementCount(n, ReductionKernelMode::OptimizedTile8));
    max_elements = std::max(max_elements, GetReduceSumWorkspaceElementCount(n, ReductionKernelMode::Optimized2Stage));
    max_elements = std::max(max_elements, GetReduceSumWorkspaceElementCount(n, ReductionKernelMode::Optimized2StageTile));
    return max_elements;
}

void BenchMode(const char* label,
               ReductionKernelMode mode,
               const float* d_input,
               float* d_output,
               int n,
               int warmup,
               int repeat,
               ReductionWorkspace workspace,
               cudaStream_t stream) {
    for (int i = 0; i < warmup; ++i) {
        LaunchReduceSumWithWorkspace(d_input, d_output, n, mode, workspace, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> ms;
    ms.reserve(repeat);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < repeat; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        LaunchReduceSumWithWorkspace(d_input, d_output, n, mode, workspace, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        ms.push_back(elapsed);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double sum = 0.0;
    for (float v : ms) sum += v;
    double avg = sum / ms.size();
    double p50 = Percentile(ms, 0.50);
    double p95 = Percentile(ms, 0.95);
    double max_v = *std::max_element(ms.begin(), ms.end());
    double gb = static_cast<double>(n) * sizeof(float) / 1e9;
    std::printf("%s: n=%d avg_ms=%.6f p50_ms=%.6f p95_ms=%.6f max_ms=%.6f effective_GBps=%.3f\n",
                label, n, avg, p50, p95, max_v, gb / (avg / 1000.0));
}


void BenchCubDeviceReduce(const char* label,
                          const float* d_input,
                          float* d_output,
                          int n,
                          int warmup,
                          int repeat,
                          void* d_temp_storage,
                          std::size_t temp_storage_bytes,
                          cudaStream_t stream) {
    for (int i = 0; i < warmup; ++i) {
        LaunchCubDeviceReduceSum(d_input, d_output, n, d_temp_storage, temp_storage_bytes, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> ms;
    ms.reserve(repeat);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < repeat; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        LaunchCubDeviceReduceSum(d_input, d_output, n, d_temp_storage, temp_storage_bytes, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        ms.push_back(elapsed);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double sum = 0.0;
    for (float v : ms) sum += v;
    double avg = sum / ms.size();
    double p50 = Percentile(ms, 0.50);
    double p95 = Percentile(ms, 0.95);
    double max_v = *std::max_element(ms.begin(), ms.end());
    double gb = static_cast<double>(n) * sizeof(float) / 1e9;
    std::printf("%s: n=%d avg_ms=%.6f p50_ms=%.6f p95_ms=%.6f max_ms=%.6f effective_GBps=%.3f\n",
                label, n, avg, p50, p95, max_v, gb / (avg / 1000.0));
}

}  // namespace

int main(int argc, char** argv) {
    int n = 1 << 24;
    int repeat = 100;
    int warmup = 20;
    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) repeat = std::atoi(argv[2]);

    std::vector<float> h_input = MakeInput(n);
    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_workspace = nullptr;
    void* d_cub_temp_storage = nullptr;
    std::size_t workspace_elements = MaxBenchmarkWorkspaceElements(n);
    std::size_t cub_temp_storage_bytes = 0;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    if (workspace_elements > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_elements * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cub_temp_storage_bytes = GetCubDeviceReduceSumTempBytes(d_input, d_output, n, stream);
    if (cub_temp_storage_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_cub_temp_storage, cub_temp_storage_bytes));
    }
    ReductionWorkspace workspace{d_workspace, workspace_elements};

    BenchCubDeviceReduce("library_cub_device_reduce", d_input, d_output, n, warmup, repeat,
                         d_cub_temp_storage, cub_temp_storage_bytes, stream);
    BenchMode("custom_cuda_atomic", ReductionKernelMode::BaselineAtomic,
              d_input, d_output, n, warmup, repeat, workspace, stream);
    BenchMode("optimized_todo", ReductionKernelMode::OptimizedTodo,
              d_input, d_output, n, warmup, repeat, workspace, stream);
    BenchMode("optimized_tile2", ReductionKernelMode::OptimizedTile2,
              d_input, d_output, n, warmup, repeat, workspace, stream);
    BenchMode("optimized_tile4", ReductionKernelMode::OptimizedTile4,
              d_input, d_output, n, warmup, repeat, workspace, stream);
    BenchMode("optimized_tile8", ReductionKernelMode::OptimizedTile8,
              d_input, d_output, n, warmup, repeat, workspace, stream);
    BenchMode("optimized_2stage", ReductionKernelMode::Optimized2Stage,
              d_input, d_output, n, warmup, repeat, workspace, stream);
    BenchMode("optimized_2stage_tile", ReductionKernelMode::Optimized2StageTile,
              d_input, d_output, n, warmup, repeat, workspace, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaStreamDestroy(stream));
    if (d_workspace) {
        CUDA_CHECK(cudaFree(d_workspace));
    }
    if (d_cub_temp_storage) {
        CUDA_CHECK(cudaFree(d_cub_temp_storage));
    }
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}

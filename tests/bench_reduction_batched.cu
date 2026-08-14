#include "kernels/cuda_check.cuh"
#include "kernels/reduction_kernel.cuh"
#include "tests/reduction_library_baselines.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

std::vector<float> MakeInput(std::size_t total_n) {
    std::vector<float> x(total_n);
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < total_n; ++i) {
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

std::size_t MaxBenchmarkWorkspaceElements(int n_per_batch) {
    std::size_t max_elements = 0;
    max_elements = std::max(max_elements, GetReduceSumWorkspaceElementCount(n_per_batch, ReductionKernelMode::BaselineAtomic));
    max_elements = std::max(max_elements, GetReduceSumWorkspaceElementCount(n_per_batch, ReductionKernelMode::OptimizedTodo));
    max_elements = std::max(max_elements, GetReduceSumWorkspaceElementCount(n_per_batch, ReductionKernelMode::OptimizedTile2));
    max_elements = std::max(max_elements, GetReduceSumWorkspaceElementCount(n_per_batch, ReductionKernelMode::OptimizedTile4));
    max_elements = std::max(max_elements, GetReduceSumWorkspaceElementCount(n_per_batch, ReductionKernelMode::OptimizedTile8));
    max_elements = std::max(max_elements, GetReduceSumWorkspaceElementCount(n_per_batch, ReductionKernelMode::Optimized2Stage));
    max_elements = std::max(max_elements, GetReduceSumWorkspaceElementCount(n_per_batch, ReductionKernelMode::Optimized2StageTile));
    return max_elements;
}

void PrintStats(const char* label,
                int n_per_batch,
                int batch,
                const std::vector<float>& ms) {
    double sum = 0.0;
    for (float v : ms) sum += v;
    double avg = sum / ms.size();
    double p50 = Percentile(ms, 0.50);
    double p95 = Percentile(ms, 0.95);
    double max_v = *std::max_element(ms.begin(), ms.end());
    double gb = static_cast<double>(n_per_batch) * static_cast<double>(batch) * sizeof(float) / 1e9;
    std::printf("%s: n_per_batch=%d batch=%d total_elements=%lld avg_ms=%.6f p50_ms=%.6f p95_ms=%.6f max_ms=%.6f effective_GBps=%.3f\n",
                label, n_per_batch, batch, static_cast<long long>(n_per_batch) * batch,
                avg, p50, p95, max_v, gb / (avg / 1000.0));
}

void BenchModeBatched(const char* label,
                      ReductionKernelMode mode,
                      const float* d_input,
                      float* d_output,
                      int n_per_batch,
                      int batch,
                      int warmup,
                      int repeat,
                      ReductionWorkspace workspace,
                      cudaStream_t stream) {
    for (int i = 0; i < warmup; ++i) {
        for (int b = 0; b < batch; ++b) {
            LaunchReduceSumWithWorkspace(d_input + static_cast<std::size_t>(b) * n_per_batch,
                                         d_output + b,
                                         n_per_batch,
                                         mode,
                                         workspace,
                                         stream);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> ms;
    ms.reserve(repeat);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < repeat; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int b = 0; b < batch; ++b) {
            LaunchReduceSumWithWorkspace(d_input + static_cast<std::size_t>(b) * n_per_batch,
                                         d_output + b,
                                         n_per_batch,
                                         mode,
                                         workspace,
                                         stream);
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        ms.push_back(elapsed);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    PrintStats(label, n_per_batch, batch, ms);
}

void BenchCubBatched(const char* label,
                     const float* d_input,
                     float* d_output,
                     int n_per_batch,
                     int batch,
                     int warmup,
                     int repeat,
                     void* d_temp_storage,
                     std::size_t temp_storage_bytes,
                     cudaStream_t stream) {
    for (int i = 0; i < warmup; ++i) {
        for (int b = 0; b < batch; ++b) {
            LaunchCubDeviceReduceSum(d_input + static_cast<std::size_t>(b) * n_per_batch,
                                     d_output + b,
                                     n_per_batch,
                                     d_temp_storage,
                                     temp_storage_bytes,
                                     stream);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> ms;
    ms.reserve(repeat);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < repeat; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int b = 0; b < batch; ++b) {
            LaunchCubDeviceReduceSum(d_input + static_cast<std::size_t>(b) * n_per_batch,
                                     d_output + b,
                                     n_per_batch,
                                     d_temp_storage,
                                     temp_storage_bytes,
                                     stream);
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        ms.push_back(elapsed);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    PrintStats(label, n_per_batch, batch, ms);
}

}  // namespace

int main(int argc, char** argv) {
    int n_per_batch = 1 << 20;
    int batch = 8;
    int repeat = 100;
    int warmup = 20;
    if (argc > 1) n_per_batch = std::atoi(argv[1]);
    if (argc > 2) batch = std::atoi(argv[2]);
    if (argc > 3) repeat = std::atoi(argv[3]);
    if (n_per_batch < 0 || batch <= 0 || repeat <= 0) {
        std::fprintf(stderr, "usage: bench_reduction_batched [n_per_batch>=0] [batch>0] [repeat>0]\n");
        return 2;
    }

    std::size_t total_elements = static_cast<std::size_t>(n_per_batch) * static_cast<std::size_t>(batch);
    std::vector<float> h_input = MakeInput(total_elements);

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_workspace = nullptr;
    void* d_cub_temp_storage = nullptr;
    std::size_t workspace_elements = MaxBenchmarkWorkspaceElements(n_per_batch);
    std::size_t cub_temp_storage_bytes = 0;

    CUDA_CHECK(cudaMalloc(&d_input, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, static_cast<std::size_t>(batch) * sizeof(float)));
    if (workspace_elements > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_elements * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cub_temp_storage_bytes = GetCubDeviceReduceSumTempBytes(d_input, d_output, n_per_batch, stream);
    if (cub_temp_storage_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_cub_temp_storage, cub_temp_storage_bytes));
    }
    ReductionWorkspace workspace{d_workspace, workspace_elements};

    BenchCubBatched("library_cub_device_reduce", d_input, d_output, n_per_batch, batch, warmup, repeat,
                    d_cub_temp_storage, cub_temp_storage_bytes, stream);
    BenchModeBatched("custom_cuda_atomic", ReductionKernelMode::BaselineAtomic,
                     d_input, d_output, n_per_batch, batch, warmup, repeat, workspace, stream);
    BenchModeBatched("optimized_todo", ReductionKernelMode::OptimizedTodo,
                     d_input, d_output, n_per_batch, batch, warmup, repeat, workspace, stream);
    BenchModeBatched("optimized_tile2", ReductionKernelMode::OptimizedTile2,
                     d_input, d_output, n_per_batch, batch, warmup, repeat, workspace, stream);
    BenchModeBatched("optimized_tile4", ReductionKernelMode::OptimizedTile4,
                     d_input, d_output, n_per_batch, batch, warmup, repeat, workspace, stream);
    BenchModeBatched("optimized_tile8", ReductionKernelMode::OptimizedTile8,
                     d_input, d_output, n_per_batch, batch, warmup, repeat, workspace, stream);
    BenchModeBatched("optimized_2stage", ReductionKernelMode::Optimized2Stage,
                     d_input, d_output, n_per_batch, batch, warmup, repeat, workspace, stream);
    BenchModeBatched("optimized_2stage_tile", ReductionKernelMode::Optimized2StageTile,
                     d_input, d_output, n_per_batch, batch, warmup, repeat, workspace, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaStreamDestroy(stream));
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
    if (d_cub_temp_storage) CUDA_CHECK(cudaFree(d_cub_temp_storage));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}

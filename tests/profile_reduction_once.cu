#include "kernels/cuda_check.cuh"
#include "kernels/reduction_kernel.cuh"
#include "tests/reduction_library_baselines.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
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

ReductionKernelMode ModeFromString(const std::string& mode) {
    if (mode == "baseline_atomic" || mode == "custom_cuda_atomic") return ReductionKernelMode::BaselineAtomic;
    if (mode == "optimized_todo") return ReductionKernelMode::OptimizedTodo;
    if (mode == "optimized_tile2") return ReductionKernelMode::OptimizedTile2;
    if (mode == "optimized_tile4") return ReductionKernelMode::OptimizedTile4;
    if (mode == "optimized_tile8") return ReductionKernelMode::OptimizedTile8;
    if (mode == "optimized_2stage") return ReductionKernelMode::Optimized2Stage;
    if (mode == "optimized_2stage_tile") return ReductionKernelMode::Optimized2StageTile;
    throw std::runtime_error("unknown mode: " + mode);
}

}  // namespace

int main(int argc, char** argv) {
    std::string mode_name = argc > 1 ? argv[1] : "optimized_tile8";
    int n = argc > 2 ? std::atoi(argv[2]) : (1 << 24);
    if (n < 0) {
        std::fprintf(stderr, "n must be non-negative\n");
        return 2;
    }

    bool use_cub = mode_name == "library_cub_device_reduce";
    ReductionKernelMode mode = use_cub ? ReductionKernelMode::OptimizedTodo : ModeFromString(mode_name);
    std::vector<float> h_input = MakeInput(n);

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_workspace = nullptr;
    void* d_cub_temp_storage = nullptr;
    std::size_t workspace_elements = use_cub ? 0 : GetReduceSumWorkspaceElementCount(n, mode);
    std::size_t cub_temp_storage_bytes = 0;

    CUDA_CHECK(cudaMalloc(&d_input, std::max(1, n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    if (workspace_elements > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_elements * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (use_cub) {
        cub_temp_storage_bytes = GetCubDeviceReduceSumTempBytes(d_input, d_output, n, stream);
        if (cub_temp_storage_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&d_cub_temp_storage, cub_temp_storage_bytes));
        }
        LaunchCubDeviceReduceSum(d_input, d_output, n, d_cub_temp_storage, cub_temp_storage_bytes, stream);
    } else {
        LaunchReduceSumWithWorkspace(d_input, d_output, n, mode,
                                     ReductionWorkspace{d_workspace, workspace_elements}, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("profile_once mode=%s n=%d result=% .9g workspace_elements=%zu cub_temp_bytes=%zu\n",
                mode_name.c_str(), n, result, workspace_elements, cub_temp_storage_bytes);

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

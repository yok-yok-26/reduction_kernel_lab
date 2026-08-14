#pragma once

#include <cuda_runtime.h>

#include <cstddef>

enum class ReductionKernelMode {
    BaselineAtomic = 0,
    OptimizedTodo = 1,
    OptimizedTile = 2,
    OptimizedTile2 = 3,
    OptimizedTile4 = 2,
    OptimizedTile8 = 4, 
    Optimized2Stage = 5,
    Optimized2StageTile = 6
};

struct ReductionWorkspace {
    float* d_temp = nullptr;
    std::size_t element_count = 0;
};

std::size_t GetReduceSumWorkspaceElementCount(int n, ReductionKernelMode mode);

void LaunchReduceSumWithWorkspace(const float* d_input,
                                  float* d_output,
                                  int n,
                                  ReductionKernelMode mode,
                                  ReductionWorkspace workspace,
                                  cudaStream_t stream);

void LaunchReduceSum(const float* d_input,
                     float* d_output,
                     int n,
                     ReductionKernelMode mode,
                     cudaStream_t stream);

#pragma once

#include "kernels/cuda_check.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

inline std::size_t GetCubDeviceReduceSumTempBytes(const float* d_input,
                                                   float* d_output,
                                                   int n,
                                                   cudaStream_t stream) {
    if (n <= 0) {
        return 0;
    }
    std::size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_bytes, d_input, d_output, n, stream));
    return temp_bytes;
}

inline void LaunchCubDeviceReduceSum(const float* d_input,
                                     float* d_output,
                                     int n,
                                     void* d_temp_storage,
                                     std::size_t temp_storage_bytes,
                                     cudaStream_t stream) {
    if (!d_input || !d_output) {
        throw std::runtime_error("LaunchCubDeviceReduceSum received null pointer");
    }
    if (n < 0) {
        throw std::runtime_error("LaunchCubDeviceReduceSum received negative n");
    }
    if (n == 0) {
        CUDA_CHECK(cudaMemsetAsync(d_output, 0, sizeof(float), stream));
        return;
    }
    if (!d_temp_storage || temp_storage_bytes == 0) {
        throw std::runtime_error("LaunchCubDeviceReduceSum requires preallocated CUB temp storage");
    }
    CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, n, stream));
}

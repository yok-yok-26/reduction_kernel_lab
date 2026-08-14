#include "kernels/reduction_kernel.cuh"

#include "kernels/cuda_check.cuh"

#include <stdexcept>
#include <cstddef>

namespace {
__global__ void ReduceSumBaselineAtomicKernel(const float* input,
                                              float* output,
                                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(output, input[idx]);
    }
}

// TODO(student): optimize this kernel.
//
// Goal:
//   Compute output[0] = sum(input[0:n]).
//
// Constraints:
//   1. Correct for n = 0, 1, blockDim-1, blockDim, blockDim+1, non-divisible n,
//      and large n.
//   2. Do not use cudaDeviceSynchronize inside the kernel path.
//   3. Keep all writes in-bounds.
//   4. You may use shared memory, warp-level primitives, multiple passes, or
//      atomics. Start simple, then improve.
//
// Learning path:
//   Round 1: block-level shared-memory reduction + one atomicAdd per block.
//   Round 2: replace the final shared-memory tail with warp shuffle reduction.
//   Round 3: reduce two or more elements per thread to improve memory throughput.

__device__ __inline__ 
float warp_down_reduce_sum(unsigned mask, float val){

    float temp;
    for (int stride = 16; stride > 0; stride>>=1)
    {
        temp = __shfl_down_sync(mask, val, stride);
        val += temp;
    }
    
    return val;
}


//////////////////////////////////
// TODO(student): two stage optimized reduction kernel using warp shuffle and temporary global memory.
//////////////////////////////////
__global__ void ReduceSumOptimizedKernelOutplace(const float* input,
                                             float* output,
                                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    __shared__ float temp_sum[32];

    float local_val = idx < n ? input[idx] : 0.0f;
    unsigned mask = __activemask();

    float local_sum = warp_down_reduce_sum(mask, local_val);
    if (lane_id == 0)
    {
        temp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        local_val = lane_id < (blockDim.x >> 5) ? temp_sum[lane_id] : 0.0f;
        local_sum = warp_down_reduce_sum(mask, local_val);
        if (lane_id == 0)
        {
            output[blockIdx.x] = local_sum;
        }
    }
}



__global__ void ReduceSumOptimizedKernelInplace(
                                             float* output,
                                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    __shared__ float temp_sum[32];

    float local_val = idx < n ? output[idx] : 0.0f;
    unsigned mask = __activemask();

    float local_sum = warp_down_reduce_sum(mask, local_val);
    if (lane_id == 0)
    {
        temp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        local_val = lane_id < (blockDim.x >> 5) ? temp_sum[lane_id] : 0.0f;
        local_sum = warp_down_reduce_sum(mask, local_val);
        if (lane_id == 0)
        {
            output[blockIdx.x] = local_sum;
        }
    }
}



//////////////////////////////////
// TODO(student): implement optimized reduction kernel using warp shuffle and multiple passes.
//////////////////////////////////
__global__ void ReduceSumOptimizedKernel(const float* input,
                                             float* output,
                                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    __shared__ float temp_sum[32];

    float local_val = idx < n ? input[idx] : 0.0f;
    unsigned mask = __activemask();

    float local_sum = warp_down_reduce_sum(mask, local_val);
    if (lane_id == 0)
    {
        temp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        local_val = lane_id < (blockDim.x >> 5) ? temp_sum[lane_id] : 0.0f;
        local_sum = warp_down_reduce_sum(mask, local_val);
        if (lane_id == 0)
        {
            atomicAdd(output, local_sum);
        }
    }
}


//////////////////////////////////
// TODO(student): implement tiled reduction kernel.
//////////////////////////////////
template <u_int8_t TILE_SIZE>
__global__ void ReduceSumOptimizedKernelTile(const float* input,
                                             float* output,
                                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    __shared__ float temp_sum[32];

    float local_val = -0.0f;
    for (int i = idx; i < n; i+=(gridDim.x*blockDim.x))
    {
        local_val += input[i];
    }
    

    unsigned mask = __activemask();

    float local_sum = warp_down_reduce_sum(mask, local_val);
    if (lane_id == 0)
    {
        temp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        local_val = lane_id < (blockDim.x >> 5) ? temp_sum[lane_id] : 0.0f;
        local_sum = warp_down_reduce_sum(mask, local_val);
        if (lane_id == 0)
        {
            atomicAdd(output, local_sum);
        }
    }
}





//////////////////////////////////
// TODO(student): two stage optimized reduction kernel using warp shuffle and tile.
//////////////////////////////////
template <u_int8_t TILE_SIZE>
__global__ void ReduceSumOptimizedKernelOutplaceTile(const float* input,
                                             float* output,
                                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    __shared__ float temp_sum[32];
    
    float local_val = 0.0f;
    for (int i = idx; i < n; i+=gridDim.x * blockDim.x)
    {
        local_val += input[i];
    }

    unsigned mask = __activemask();

    float local_sum = warp_down_reduce_sum(mask, local_val);
    if (lane_id == 0)
    {
        temp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        local_val = lane_id < (blockDim.x >> 5) ? temp_sum[lane_id] : 0.0f;
        local_sum = warp_down_reduce_sum(mask, local_val);
        if (lane_id == 0)
        {
            output[blockIdx.x] = local_sum;
        }
    }
}





}  // namespace

std::size_t GetReduceSumWorkspaceElementCount(int n, ReductionKernelMode mode) {
    if (n <= 0) {
        return 0;
    }

    constexpr int kBlock = 256;
    constexpr int kBlockTile = 256;
    constexpr int kTile8 = 8;

    if (mode == ReductionKernelMode::Optimized2Stage) {
        return static_cast<std::size_t>((n + kBlock - 1) / kBlock);
    }
    if (mode == ReductionKernelMode::Optimized2StageTile) {
        return static_cast<std::size_t>((n + (kBlockTile * kTile8) - 1) / (kBlockTile * kTile8));
    }
    return 0;
}

void LaunchReduceSumWithWorkspace(const float* d_input,
                                  float* d_output,
                                  int n,
                                  ReductionKernelMode mode,
                                  ReductionWorkspace workspace,
                                  cudaStream_t stream) {
    if (!d_input || !d_output) {
        throw std::runtime_error("LaunchReduceSum received null pointer");
    }
    if (n < 0) {
        throw std::runtime_error("LaunchReduceSum received negative n");
    }

    CUDA_CHECK(cudaMemsetAsync(d_output, 0, sizeof(float), stream));

    constexpr int kBlock = 256;
    int grid = (n + kBlock - 1) / kBlock;
    if (grid == 0) {
        CUDA_KERNEL_CHECK();
        return;
    }


    constexpr int kBlockTile = 256;
    constexpr int kTile2 = 2;
    constexpr int kTile4 = 4;
    constexpr int kTile8 = 8;
    int grid_tile2 = (n + (kBlockTile * kTile2) - 1) / (kBlockTile * kTile2);
    int grid_tile4 = (n + (kBlockTile * kTile4) - 1) / (kBlockTile * kTile4);
    int grid_tile8 = (n + (kBlockTile * kTile8) - 1) / (kBlockTile * kTile8);

    if (mode == ReductionKernelMode::BaselineAtomic) {
        ReduceSumBaselineAtomicKernel<<<grid, kBlock, 0, stream>>>(d_input, d_output, n);
    } else if (mode == ReductionKernelMode::OptimizedTodo) {
        ReduceSumOptimizedKernel<<<grid, kBlock, 0, stream>>>(d_input, d_output, n);
    } else if (mode == ReductionKernelMode::OptimizedTile2) {
        ReduceSumOptimizedKernelTile<kTile2><<<grid_tile2, kBlockTile, 0, stream>>>(d_input, d_output, n);
    } else if (mode == ReductionKernelMode::OptimizedTile ||
               mode == ReductionKernelMode::OptimizedTile4) {
        ReduceSumOptimizedKernelTile<kTile4><<<grid_tile4, kBlockTile, 0, stream>>>(d_input, d_output, n);
    } else if (mode == ReductionKernelMode::OptimizedTile8) {
        ReduceSumOptimizedKernelTile<kTile8><<<grid_tile8, kBlockTile, 0, stream>>>(d_input, d_output, n);
    } else if (mode == ReductionKernelMode::Optimized2Stage) {
        std::size_t required_workspace = GetReduceSumWorkspaceElementCount(n, mode);
        if (!workspace.d_temp || workspace.element_count < required_workspace) {
            throw std::runtime_error("Optimized2Stage requires a preallocated reduction workspace");
        }
        float* d_tmp = workspace.d_temp;

        int partial_count = grid;
        ReduceSumOptimizedKernelOutplace<<<grid, kBlock, 0, stream>>>(d_input, d_tmp, n);
        CUDA_KERNEL_CHECK();

        while (partial_count > 1) {
            int next_grid = (partial_count + kBlock - 1) / kBlock;
            ReduceSumOptimizedKernelInplace<<<next_grid, kBlock, 0, stream>>>(d_tmp, partial_count);
            CUDA_KERNEL_CHECK();
            partial_count = next_grid;
        }
        CUDA_CHECK(cudaMemcpyAsync(d_output, d_tmp, sizeof(float), cudaMemcpyDeviceToDevice, stream));
    } else if (mode == ReductionKernelMode::Optimized2StageTile) {
        std::size_t required_workspace = GetReduceSumWorkspaceElementCount(n, mode);
        if (!workspace.d_temp || workspace.element_count < required_workspace) {
            throw std::runtime_error("Optimized2StageTile requires a preallocated reduction workspace");
        }
        float* d_tmp = workspace.d_temp;

        int partial_count = grid_tile8;
        ReduceSumOptimizedKernelOutplaceTile<kTile8><<<grid_tile8, kBlockTile, 0, stream>>>(d_input, d_tmp, n);
        CUDA_KERNEL_CHECK();

        while (partial_count > 1) {
            int next_grid = (partial_count + kBlock - 1) / kBlock;
            ReduceSumOptimizedKernelInplace<<<next_grid, kBlock, 0, stream>>>(d_tmp, partial_count);
            CUDA_KERNEL_CHECK();
            partial_count = next_grid;
        }
        CUDA_CHECK(cudaMemcpyAsync(d_output, d_tmp, sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
    
    
    CUDA_KERNEL_CHECK();

#ifdef DEBUG_CUDA_SYNC
    CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
}


void LaunchReduceSum(const float* d_input,
                     float* d_output,
                     int n,
                     ReductionKernelMode mode,
                     cudaStream_t stream) {
    LaunchReduceSumWithWorkspace(d_input, d_output, n, mode, ReductionWorkspace{}, stream);
}

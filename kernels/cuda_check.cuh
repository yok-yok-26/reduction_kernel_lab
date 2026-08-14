#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(err__));   \
            std::abort();                                                  \
        }                                                                  \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                \
    do {                                                                   \
        cudaError_t err__ = cudaGetLastError();                            \
        if (err__ != cudaSuccess) {                                        \
            std::fprintf(stderr, "Kernel launch error %s:%d: %s\n",       \
                         __FILE__, __LINE__, cudaGetErrorString(err__));   \
            std::abort();                                                  \
        }                                                                  \
    } while (0)

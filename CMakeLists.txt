cmake_minimum_required(VERSION 3.18)
project(reduction_kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

option(DEBUG_CUDA_SYNC "Synchronize after kernel launches in debug paths" OFF)

add_library(reduction_kernel STATIC
  kernels/reduction_kernel.cu
)
target_include_directories(reduction_kernel PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(reduction_kernel PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G;-O0;-g>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
)
if(DEBUG_CUDA_SYNC)
  target_compile_definitions(reduction_kernel PUBLIC DEBUG_CUDA_SYNC=1)
endif()

add_library(cpu_reference STATIC
  reference/cpu_reduce.cpp
)
target_include_directories(cpu_reference PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(test_reduction_correctness
  tests/test_reduction_correctness.cu
)
target_link_libraries(test_reduction_correctness PRIVATE
  reduction_kernel
  cpu_reference
)

add_executable(bench_reduction
  tests/bench_reduction.cu
)
target_link_libraries(bench_reduction PRIVATE
  reduction_kernel
  cpu_reference
)

add_executable(profile_reduction_once
  tests/profile_reduction_once.cu
)
target_link_libraries(profile_reduction_once PRIVATE
  reduction_kernel
)

add_executable(bench_reduction_batched
  tests/bench_reduction_batched.cu
)
target_link_libraries(bench_reduction_batched PRIVATE
  reduction_kernel
)

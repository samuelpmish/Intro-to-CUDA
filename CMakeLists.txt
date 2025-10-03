cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED CMAKE_BUILD_TYPE)
  message(WARNING "CMAKE_BUILD_TYPE not defined.  Defaulting to release.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type: Debug;Release;MinSizeRel;RelWithDebInfo")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(WARNING "CMAKE_CUDA_ARCHITECTURES not defined.  Defaulting to native")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda_exercises LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit)

#add_compile_options("-Xptxas")
#add_compile_options("-v")
#add_compile_options("-lineinfo") 

include_directories(common)

add_executable(hello_world 01_kernel_basics/hello_world.cu)
add_executable(launch_config 01_kernel_basics/launch_config.cu)
add_executable(thread_divergence 01_kernel_basics/thread_divergence.cu)
add_executable(saxpy_cpu 01_kernel_basics/saxpy_cpu.cu)
add_executable(saxpy 01_kernel_basics/saxpy_unfinished.cu)
add_executable(tail_effect 01_kernel_basics/tail_effect.cu)
add_executable(laplace_operator_cpu 01_kernel_basics/laplace_operator_cpu.cu)
add_executable(laplace_operator 01_kernel_basics/laplace_operator.cu)

add_executable(host_device_annotations 01_kernel_basics/host_device_annotations.cu)
target_compile_options(host_device_annotations PUBLIC "--expt-relaxed-constexpr")
target_compile_options(host_device_annotations PUBLIC "-extended-lambda")

add_executable(memory_spaces 02_memory/memory_spaces.cu)
add_executable(cpu_cpu_bandwidth 02_memory/cpu_cpu_bandwidth.cu)
add_executable(cpu_gpu_bandwidth 02_memory/cpu_gpu_bandwidth.cu)
add_executable(gpu_gpu_bandwidth 02_memory/gpu_gpu_bandwidth.cu)

add_executable(cpu_reduction 03_parallel_reduction/cpu_reduction.cu)
add_executable(gpu_reduction 03_parallel_reduction/gpu_reduction.cu)
add_executable(gpu_reduction_shmem 03_parallel_reduction/gpu_reduction_shmem.cu)
add_executable(gpu_reduction_thrust 03_parallel_reduction/gpu_reduction_thrust.cu)

add_executable(scatter_gather scatter_gather.cu)

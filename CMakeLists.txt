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

# these flags turn on verbose output when compiling kernels
# (e.g. register usage, shared memory footprint, etc)
#add_compile_options("-Xptxas")
#add_compile_options("-v")

# this flag enables kernel debugging 
#add_compile_options("-G")

# this flag enables line of code associations 
# from tools like compute-sanitizer
add_compile_options("-lineinfo") 

# these flags enable generation of detailed reports of 
# where compilation time in spent in each translation unit
#        note: g++ does not support -ftime-trace!
#add_compile_options(
#    $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-ftime-trace> 
#    $<$<COMPILE_LANGUAGE:CUDA>:-fdevice-time-trace=->
#)

include_directories(common)

file(GLOB sources
  01_kernel_basics/*.cu
  01_kernel_basics/*.cpp
  02_memory/*.cu
  03_parallel_reduction/*.cu
  04_streams/*.cu
  05_bugs/*.cu
  06_profiling/unstructured_mesh/*.cu
)

foreach(filename ${sources})
  get_filename_component(testname ${filename} NAME_WE)
  add_executable(${testname} ${filename})
endforeach()

target_compile_options(host_device_annotations PUBLIC "--expt-relaxed-constexpr")
target_compile_options(host_device_annotations PUBLIC "-extended-lambda")

add_executable(conjugate_gradient_0 
  06_profiling/conjugate_gradient_0/vector.cu
  06_profiling/conjugate_gradient_0/poisson.cu
)

add_executable(conjugate_gradient_1 
  06_profiling/conjugate_gradient_1/vector.cu
  06_profiling/conjugate_gradient_1/poisson.cu
)

add_executable(conjugate_gradient_2 
  06_profiling/conjugate_gradient_2/vector.cu
  06_profiling/conjugate_gradient_2/poisson.cu
)

add_executable(box_blur 06_profiling/box_blur/box_blur.cu)
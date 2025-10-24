// Goal: modify this program to go from executing 64 kernels sequentially
//       to running them on `n` different streams. 
//   - Measure the performance for different numbers of streams (1, 2, 4, ... 64) 
//       and comment on the results. 
//   - Try visualizing the kernel execution with NSight systems: 
//       $ nsys profile ./streams_for_concurrent_kernels_monolithic
//       and opening the generated report with NSight systems on your local machine
//
// For reference, an individual stream can be created / destroyed with
//
//   cudaStream_t stream;
//   cudaStreamCreate(&stream);
//
//   ...
//
//   cudaStreamDestroy(stream);
//
// and kernels can be launched on a specific stream by passing it as the 4th argument in the kernel launch:
//
//   my_kernel<<< grid, block, shmem, stream >>>(args ...);
//                                    ^^^^^^
// if the kernel doesn't use dynamic shared memory, then leave shmem = 0

#include <vector>
#include <iostream>

#include "timer.hpp"

__global__ void kernel(float * data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // some expensive calculation
        for (int k = 0; k < 1000; k++) {
            int id = (i + 128 * k) % n;
            data[id] = sin(data[id]);
        }
    }
}

int main() {

    int n = 1 << 12;
    int num_buffers = 64;

    std::vector< float * > d_data(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        cudaMalloc(&d_data[i], n * sizeof(float));
    }

    cudaDeviceSynchronize();

    timer stopwatch;

    stopwatch.start();

    int block = 256;
    int grid = n / block;

    for (int i = 0; i < num_buffers; i++) {
        kernel<<< grid, block >>>(d_data[i], n);
    }

    cudaDeviceSynchronize();

    stopwatch.stop();

    std::cout << stopwatch.elapsed() * 1000.0f << " ms" << std::endl;

    for (int i = 0; i < num_buffers; i++) {
        cudaFree(d_data[i]);
    }

}

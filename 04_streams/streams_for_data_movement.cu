// Goal: modify this program to go from executing the H2D / Kernel / D2H on a single stream
//       to using `n` different streams.
//   - Measure the performance for different numbers of streams (1, 2, 4, ... 64) 
//       and comment on the results. 
//   - Try making the kernel duration of the kernel longer or shorter by a factor of ten 
//       (by changing the number of iterations in the kernel). How does the number of streams
//       runtimes in each case?
//   - Try visualizing the kernel execution with NSight systems: 
//       $ nsys profile ./streams_for_concurrent_kernels_monolithic
//       and opening the generated report with NSight systems on your local machine
//
// For reference, streams can be created / destroyed with:
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
//   my_kernel<<< grid, block, shmem, stream>>>(args ...);
//                                    ^^^^^^
// 
// Allocate the CPU buffers with cudaMallocHost, and copy data to and from the host with 
//                                             
//   cudaMemcpyAsync(dst, src, nbytes, kind, stream) to
//                                           ^^^^^^
#include <vector>
#include <iostream>

#include "timer.hpp"

__global__ void kernel(float * data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float value = data[i];
        for (int k = 0; k < 100; k++) {
            value = sin(value); // some expensive calculation
        }
        data[i] = value;
    }
}

int main() {

    int n = 1 << 24;

    float * h_data;
    float * d_data;

    cudaMallocHost(&h_data, n * sizeof(float));
    cudaMalloc(&d_data, n * sizeof(float));

    for (int i = 0; i < n; i++) {
        h_data[i] = i;
    }

    timer stopwatch;

    stopwatch.start();

    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = n / block;
    kernel<<< grid, block >>>(d_data, n);

    cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    stopwatch.stop();

    std::cout << stopwatch.elapsed() * 1000.0f << " ms" << std::endl;

    cudaFree(d_data);
    cudaFreeHost(h_data);

}
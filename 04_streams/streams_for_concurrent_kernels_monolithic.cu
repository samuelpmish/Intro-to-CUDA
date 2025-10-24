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

    float * d_data_monolithic;
    cudaMalloc(&d_data_monolithic, n * num_buffers * sizeof(float));

    cudaDeviceSynchronize();

    timer stopwatch;

    stopwatch.start();

    int block = 256;
    int grid = n * num_buffers / block;
    kernel<<< grid, block >>>(d_data_monolithic, n * num_buffers);
    cudaDeviceSynchronize();

    stopwatch.stop();

    std::cout << stopwatch.elapsed() * 1000.0f << " ms" << std::endl;

    cudaFree(d_data_monolithic);

}
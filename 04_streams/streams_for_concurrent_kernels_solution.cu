#include <vector>
#include <iostream>

#include "timer.hpp"

__global__ void kernel(float * data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // some expensive calculation
        for (int k = 0; k < 100; k++) {
            int id = (i + 128 * k) % n;
            data[id] = sin(data[id]);
        }
    }
}

int main() {

    int n = 1 << 12;
    int num_buffers = 64;
    int num_streams = 16;

    std::vector< float * > d_data(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        cudaMalloc(&d_data[i], n * sizeof(float));
    }

    std::vector< cudaStream_t > stream(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&stream[i]);
    }

    timer stopwatch;

    stopwatch.start();
    {
        int block = 256;
        int grid = n / block;

        for (int i = 0; i < num_buffers; i++) {
            kernel<<< grid, block, 0, stream[i % num_streams] >>>(d_data[i], n);
        }

        cudaDeviceSynchronize();
    }
    stopwatch.stop();

    std::cout << stopwatch.elapsed() * 1000.0f << " ms" << std::endl;

    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(stream[i]);
    }

    for (int i = 0; i < num_buffers; i++) {
        cudaFree(d_data[i]);
    }

}
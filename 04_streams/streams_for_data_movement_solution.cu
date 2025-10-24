#include <vector>
#include <iostream>

#include "timer.hpp"

__global__ void kernel(float * data, int n, int kmax) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float value = data[i];
        for (int k = 0; k < kmax; k++) {
            value = sin(value); // some expensive calculation
        }
        data[i] = value;
    }
}

int main() {

    int kmax = 100;
    int n = 1 << 24;
    int num_streams = 16;

    int chunk_size = n / num_streams;

    float * h_data;
    float * d_data;

    cudaMallocHost(&h_data, n * sizeof(float));
    cudaMalloc(&d_data, n * sizeof(float));

    for (int i = 0; i < n; i++) {
        h_data[i] = i;
    }

    std::vector< cudaStream_t > stream(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&stream[i]);
    }

    timer stopwatch;

    stopwatch.start();

    // asynchronously transfer H2D, launch a kernel, and transfer D2H
    for (int i = 0; i < num_streams; i++) {

        int offset = i * (n / num_streams);
        cudaMemcpyAsync(d_data + offset, 
                        h_data + offset, 
                        chunk_size * sizeof(float),
                        cudaMemcpyHostToDevice, 
                        stream[i]);

        int block = 256;
        int grid = chunk_size / block;
        kernel<<< grid, block, 0, stream[i]>>>(d_data + offset, chunk_size, kmax);

        cudaMemcpyAsync(h_data + offset, 
                        d_data + offset, 
                        chunk_size * sizeof(float),
                        cudaMemcpyDeviceToHost, 
                        stream[i]);

    }

    cudaDeviceSynchronize();

    stopwatch.stop();

    std::cout << stopwatch.elapsed() * 1000.0f << " ms" << std::endl;

    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(stream[i]);
    }

    cudaFree(d_data);
    cudaFreeHost(h_data);

}
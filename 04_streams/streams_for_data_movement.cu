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
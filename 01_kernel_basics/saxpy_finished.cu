#include <iostream>

#include "timer.hpp"

__global__ void saxpy(const float a, const float * x, float * y) {
    //int i = threadIdx.x + blockIdx.x * blockDim.x;
    int i = blockIdx.x + threadIdx.x * gridDim.x;
    y[i] = a * x[i] + y[i]; 
    float tmp = x[i];
    float tmp2 = y[i];
    y[i] = a * tmp + tmp2;

}

int main() {

    // we'll stick to powers of two so everything divides evenly
    int n = 1 << 22;

    int threads_per_block = 256;
    int blocks_per_grid = n / threads_per_block;

    float a = 1.0;
    float * x;
    float * y;

    // allocate memory for the vectors
    cudaMalloc(&x, sizeof(float) * n);
    cudaMalloc(&y, sizeof(float) * n);

    timer stopwatch;

    stopwatch.start();
    saxpy<<< blocks_per_grid, threads_per_block >>>(a, x, y);
    cudaDeviceSynchronize();
    stopwatch.stop();

    float time = stopwatch.elapsed();
    uint32_t num_bytes = n * sizeof(float) * 3; // 2 reads + 1 write

    std::cout << "time: " << time * 1000.0f << " ms " << std::endl;
    std::cout << "effective memory bandwidth: " << (num_bytes / time) * 1.0e-9f << " GB/s " << std::endl;

    // deallocate memory for the vectors
    cudaFree(x);
    cudaFree(y);

}
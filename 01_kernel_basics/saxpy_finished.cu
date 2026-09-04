#include <vector>
#include <iostream>

#include "timer.hpp"

__global__ void saxpy(const float a, const float * x, float * y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    //int i = blockIdx.x + threadIdx.x * gridDim.x;
    y[i] = a * x[i] + y[i];
}

int main() {

    // we'll stick to powers of two so everything divides evenly
    int n = 1 << 24;

    int threads_per_block = 256;
    int blocks_per_grid = n / threads_per_block;

    float a = 1.0;
    float y0 = 2.0;
    float * x;
    float * y;

    // allocate memory for the vectors
    cudaMalloc(&x, sizeof(float) * n);
    cudaMalloc(&y, sizeof(float) * n);

    // don't time first kernel
    //   note: this runs before the buffers are initialized below, so that it
    //         can't disturb the values the timed kernel operates on
    saxpy<<< 1, 1 >>>(a, x, y);
    cudaDeviceSynchronize();

    // x counts up and y is constant, so we know what the answer should be
    std::vector< float > h_x(n);
    std::vector< float > h_y(n, y0);
    for (int i = 0; i < n; i++) { h_x[i] = float(i); }

    cudaMemcpy(x, &h_x[0], sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y, &h_y[0], sizeof(float) * n, cudaMemcpyHostToDevice);

    timer stopwatch;

    stopwatch.start();
    saxpy<<< blocks_per_grid, threads_per_block >>>(a, x, y);
    cudaDeviceSynchronize();
    stopwatch.stop();

    float time = stopwatch.elapsed();
    uint32_t num_bytes = n * sizeof(float) * 3; // 2 reads + 1 write

    std::cout << "time: " << time * 1000.0f << " ms " << std::endl;
    std::cout << "effective memory bandwidth: " << (num_bytes / time) * 1.0e-9f << " GB/s " << std::endl;

    // copy the results back and check them (outside the timed section)
    cudaMemcpy(&h_y[0], y, sizeof(float) * n, cudaMemcpyDeviceToHost);

    int num_errors = 0;
    for (int i = 0; i < n; i++) {
        // a is 1, so this arithmetic is exact in fp32 and we can compare directly
        if (h_y[i] != a * h_x[i] + y0) { num_errors++; }
    }

    if (num_errors == 0) {
        std::cout << "verification: all " << n << " values correct" << std::endl;
    } else {
        std::cout << "verification: " << num_errors << " of " << n << " values are wrong!" << std::endl;
    }

    // deallocate memory for the vectors
    cudaFree(x);
    cudaFree(y);

}

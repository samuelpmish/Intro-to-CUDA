#include <vector>
#include <iostream>

#include "timer.hpp"

__global__ void saxpy(const float a, const float * x, float * y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    //int i = blockIdx.x + threadIdx.x * gridDim.x;
    y[i] = a * x[i] + y[i];
}

int main() {

    int n = 1 << 24;

    float a = 1.0;
    float y0 = 2.0;

    // x counts up and y is constant, so we know what the answer should be
    std::vector< float > h_x(n);
    std::vector< float > h_y(n);
    for (int i = 0; i < n; i++) { h_x[i] = float(i); }

for (int threads_per_block = 4; threads_per_block <= 1024; threads_per_block <<= 1) {

    int blocks_per_grid = n / threads_per_block;

    float * x;
    float * y;

    // allocate memory for the vectors
    cudaMalloc(&x, sizeof(float) * n);
    cudaMalloc(&y, sizeof(float) * n);

    // don't time the first launch
    //   note: this runs before the buffers are initialized below, so that it
    //         can't disturb the values the timed kernel operates on
    saxpy<<< 1, threads_per_block >>>(a, x, y);
    cudaDeviceSynchronize();

    for (int i = 0; i < n; i++) { h_y[i] = y0; }
    cudaMemcpy(x, &h_x[0], sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y, &h_y[0], sizeof(float) * n, cudaMemcpyHostToDevice);

    timer stopwatch;

    stopwatch.start();
    saxpy<<< blocks_per_grid, threads_per_block >>>(a, x, y);
    cudaDeviceSynchronize();
    stopwatch.stop();

    float time = stopwatch.elapsed();
    uint32_t num_bytes = n * sizeof(float) * 3; // 2 reads + 1 write

    // copy the results back and check them (outside the timed section)
    cudaMemcpy(&h_y[0], y, sizeof(float) * n, cudaMemcpyDeviceToHost);

    int num_errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_y[i] != a * h_x[i] + y0) { num_errors++; }
    }

    std::cout << threads_per_block << " " <<  (num_bytes / time) * 1.0e-9f;
    if (num_errors != 0) { std::cout << "   <-- " << num_errors << " values are wrong!"; }
    std::cout << std::endl;

    // deallocate memory for the vectors
    cudaFree(x);
    cudaFree(y);

}

}

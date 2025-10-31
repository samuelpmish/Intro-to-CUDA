#include "timer.hpp"

#include <iostream>

__global__ void empty_kernel() {
    // no-op
}

int main() {
    int n = 1000;

    empty_kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    timer stopwatch;

    stopwatch.start();
    for (int i = 0; i < n; i++) {
        empty_kernel<<<1,1>>>();
    }
    cudaDeviceSynchronize();
    stopwatch.stop();

    std::cout << "launched " << n << " kernels in ";
    std::cout << stopwatch.elapsed() * 1000.0 << " ms" << std::endl;
}
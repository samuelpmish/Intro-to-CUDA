#include <cstdio>
#include <vector>
#include <cstring>
#include <iostream>

#include "timer.hpp"

static constexpr int n = 10000000;

int main() {

    uint32_t num_iterations = 10;
    uint32_t num_bytes = n * sizeof(double);

    std::vector< double > h_data(n);

    double * d_data[2];
    cudaMalloc(&d_data[0], num_bytes);
    cudaMalloc(&d_data[1], num_bytes);

    float time_ms = 1000.0f * time([&](){
        for (int i = 0; i < num_iterations; i++) {
            cudaMemcpy(d_data[0], d_data[1], num_bytes, cudaMemcpyDeviceToDevice);
        }
        cudaDeviceSynchronize();
    });

    float to_GBps = 1.0e-6f;
    std::cout << "copied " << 2 * num_bytes * num_iterations << " total bytes in " << time_ms << " ms" << std::endl; 
    std::cout << "effective bandwidth: " << (2 * num_bytes * num_iterations / time_ms) * to_GBps << " GB/s" << std::endl;

    cudaFree(d_data[0]);
    cudaFree(d_data[1]);

}
#include <cstdio>
#include <vector>
#include <cstring>
#include <iostream>

#include "timer.hpp"
#include "pinned_allocator.hpp"

static constexpr int n = 100000;

int main() {

    uint32_t num_iterations = 10;
    uint32_t num_bytes = n * sizeof(double);

    std::vector< double > h_data(n);
    //std::vector< double, pinned_allocator<double> > h_data(n);

    double * d_data;
    cudaMalloc(&d_data, num_bytes);

    float time_ms = 1000.0f * time([&](){
        for (int i = 0; i < num_iterations; i++) {
            cudaMemcpy(d_data, &h_data[0], num_bytes, cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();
    });

    float to_GBps = 1.0e-6f;
    std::cout << "copied " << 2 * num_bytes * num_iterations << " total bytes in " << time_ms << " ms" << std::endl; 
    std::cout << "effective bandwidth: " << (2 * num_bytes * num_iterations / time_ms) * to_GBps << " GB/s" << std::endl;

    cudaFree(d_data);

}
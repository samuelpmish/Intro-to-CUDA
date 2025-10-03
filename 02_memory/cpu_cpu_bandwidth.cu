#include <cstdio>
#include <vector>
#include <cstring>
#include <iostream>

#include "timer.hpp"

static constexpr int n = 10000000;

int main() {

    uint32_t num_iterations = 10;
    uint32_t num_bytes = n * sizeof(double);

    std::vector< double > h_data[2] = {
        std::vector<double>(n),
        std::vector<double>(n)
    };

    float time_ms = 1000.0f * time([&](){
        for (int i = 0; i < num_iterations; i++) {
            std::memcpy(&h_data[0][0], &h_data[1][0], num_bytes);
        }
    });

    float to_GBps = 1.0e-6f;
    std::cout << "copied " << 2 * num_bytes * num_iterations << " total bytes in " << time_ms << " ms" << std::endl; 
    std::cout << "effective bandwidth: " << (2 * num_bytes * num_iterations / time_ms) * to_GBps << " GB/s" << std::endl;

}
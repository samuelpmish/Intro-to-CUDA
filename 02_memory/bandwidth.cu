#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "timer.hpp"
#include "pinned_allocator.hpp"

// some typical numbers
//
//   cpu <-> cpu    system DRAM      (~50-100 GB/s on a modern desktop)
//   cpu <-> gpu       PCIe            (15.75 GB/s at gen 3 x16,
//                                      31.5  GB/s at gen 4 x16,
//                                      63.0  GB/s at gen 5 x16)
//   gpu <-> gpu    device DRAM        (~1800 GB/s on an RTX 5090)

static constexpr uint64_t num_iterations = 10;

struct result {
    std::string label;
    double time_ms;
    double GBps;
};

template < typename callable >
result measure(std::string label, uint64_t bytes_per_copy, callable copy) {

    // don't time the first copy: it pays for CUDA context setup,
    // first-touch page faults, and so on
    copy();
    cudaDeviceSynchronize();

    double time_ms = 1000.0 * time([&](){
        for (uint64_t i = 0; i < num_iterations; i++) { 
            copy(); 
        }
        cudaDeviceSynchronize();
    });

    return result{ label, time_ms, (bytes_per_copy * num_iterations / time_ms) * 1.0e-6 };

}

int run(uint64_t n) {

    uint64_t num_bytes = n * sizeof(double);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "device: " << prop.name << std::endl;
    std::cout << "buffer size: " << num_bytes << " bytes (" << n << " doubles)" << std::endl;
    std::cout << "copies per test: " << num_iterations << std::endl;
    std::cout << std::endl;

    // two host buffers in ordinary, pageable memory
    std::vector< double > pageable_src(n);
    std::vector< double > pageable_dst(n);

    // one host buffer in page-locked ("pinned") memory. the driver can DMA
    // straight out of this, instead of first staging the data through an
    // internal pinned buffer of its own
    std::vector< double, pinned_allocator<double> > pinned(n);

    double * d_data[2];
    for (int i = 0; i < 2; i++) {
        auto error = cudaMalloc(&d_data[i], num_bytes);
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc: ") + cudaGetErrorString(error));
        }
    }

    std::vector< result > results;

    results.push_back(measure("cpu -> cpu  (memcpy)", 2 * num_bytes, [&](){
        std::memcpy(&pageable_dst[0], &pageable_src[0], num_bytes);
    }));

    results.push_back(measure("cpu -> gpu  (pageable)", num_bytes, [&](){
        cudaMemcpy(d_data[0], &pageable_src[0], num_bytes, cudaMemcpyHostToDevice);
    }));

    results.push_back(measure("cpu -> gpu  (pinned)", num_bytes, [&](){
        cudaMemcpy(d_data[0], &pinned[0], num_bytes, cudaMemcpyHostToDevice);
    }));

    results.push_back(measure("gpu -> cpu  (pageable)", num_bytes, [&](){
        cudaMemcpy(&pageable_dst[0], d_data[0], num_bytes, cudaMemcpyDeviceToHost);
    }));

    results.push_back(measure("gpu -> cpu  (pinned)", num_bytes, [&](){
        cudaMemcpy(&pinned[0], d_data[0], num_bytes, cudaMemcpyDeviceToHost);
    }));

    results.push_back(measure("gpu -> gpu  (cudaMemcpy)", 2 * num_bytes, [&](){
        cudaMemcpy(d_data[1], d_data[0], num_bytes, cudaMemcpyDeviceToDevice);
    }));

    std::cout << std::left  << std::setw(26) << "test";
    std::cout << std::right << std::setw(12) << "time (ms)";
    std::cout << std::right << std::setw(14) << "bandwidth" << std::endl;
    std::cout << std::string(52, '-') << std::endl;

    for (const result & r : results) {
        std::cout << std::left  << std::setw(26) << r.label;
        std::cout << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.time_ms;
        std::cout << std::right << std::setw(9)  << std::fixed << std::setprecision(2) << r.GBps << " GB/s" << std::endl;
    }

    std::cout << std::endl;

    cudaFree(d_data[0]);
    cudaFree(d_data[1]);

    return 0;

}

int main(int argc, char * argv[]) {

    // the number of doubles in each buffer
    uint64_t n = 1 << 24;

    if (argc > 1) {
        char * end;
        unsigned long long value = std::strtoull(argv[1], &end, 10);
        if (*end != '\0' || value == 0) {
            std::cerr << "usage: " << argv[0] << " [n]" << std::endl;
            std::cerr << "  n: the number of doubles in each buffer (default: " << n << ")" << std::endl;
            std::cerr << "     this needs 3n doubles of host memory and 2n of device memory" << std::endl;
            return 1;
        }
        n = value;
    }

    try {
        return run(n);
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << std::endl;
        std::cerr << "(try a smaller value of n)" << std::endl;
        return 1;
    }

}

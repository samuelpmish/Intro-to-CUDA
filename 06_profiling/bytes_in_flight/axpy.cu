#include "timer.hpp"

#include <vector>
#include <iostream>

template < typename scalar_t >
__global__ void axpy(scalar_t a, const scalar_t * x, scalar_t * y, int n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        y[tid] = a * x[tid] + y[tid];
    }
}

template < typename data_t >
void run_test(int n) {

    timer stopwatch;

    data_t * d_a;
    data_t * d_b;

    cudaMalloc(&d_a, sizeof(data_t) * n);
    cudaMalloc(&d_b, sizeof(data_t) * n);

    std::vector< data_t > h_a(n, data_t{1});
    std::vector< data_t > h_b(n, data_t{2});

    cudaMemcpy(d_a, &h_a[0], sizeof(data_t) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b[0], sizeof(data_t) * n, cudaMemcpyHostToDevice);

    axpy<<<1,1>>>(data_t{3}, d_a, d_b, 0);
    cudaDeviceSynchronize();

    stopwatch.start();
    int block = 64;
    int grid = (n + block - 1) / block;
    axpy<<<grid,block>>>(data_t{3}, d_a, d_b, n);
    cudaDeviceSynchronize();
    stopwatch.stop();

    uint64_t bytes = n * sizeof(data_t) * 3;
    std::cout << stopwatch.elapsed() * 1000.0f << " ms, ";
    std::cout << (bytes / stopwatch.elapsed()) / 1.0e09 << " GB/s" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);

}

int main() {

    std::cout << "fp32: "; run_test< float >(1 << 24);
    std::cout << "fp64: "; run_test< double >(1 << 24);
    std::cout << "  u8: "; run_test< uint8_t >(1 << 24);
    std::cout << " u16: "; run_test< uint16_t >(1 << 24);
    std::cout << " u32: "; run_test< uint32_t >(1 << 24);
    std::cout << " u64: "; run_test< uint64_t >(1 << 24);

}
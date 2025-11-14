#include "chunk.hpp"
#include "timer.hpp"

#include <vector>
#include <iostream>

__global__ void regular_axpy(double a, const double * x, double * y){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    y[tid] = a * x[tid] + y[tid];
}

__global__ void modified_axpy(double a, const double * x, double * y, int kmax){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double tmp = a * x[tid] + y[tid];
    for (int k = 0; k < kmax; k++) {
        tmp = exp(-tmp) + 1.2;
    }
    y[tid] = tmp;
}

int main() {

    int n = 1 << 22;

    double * d_a;
    double * d_b;

    cudaMalloc(&d_a, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * n);

    std::vector< double > h_a(n, double{1});
    std::vector< double > h_b(n, double{2});

    cudaMemcpy(d_a, &h_a[0], sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b[0], sizeof(double) * n, cudaMemcpyHostToDevice);

    regular_axpy<<<1,1>>>(double{3}, d_a, d_b);
    modified_axpy<<<1,1>>>(double{3}, d_a, d_b, 0);
    cudaDeviceSynchronize();

    int block = 256;
    int grid = (n + block - 1) / block;
    uint64_t bytes = n * sizeof(double) * 3;


    float regular_time_ms = kernel_time_in_ms([&](){
        regular_axpy<<<grid,block>>>(double{3}, d_a, d_b);
    });
    std::cout << "regular axpy: " << regular_time_ms << " ms, ";
    std::cout << (bytes / regular_time_ms) * 1.0e-6 << " GB/s" << std::endl;

    for (int kmax = 0; kmax < 16; kmax++) {
        float modified_time_ms = kernel_time_in_ms([&](){
            modified_axpy<<<grid,block>>>(double{3}, d_a, d_b, kmax);
        });
        std::cout << "modified (kmax = " << kmax << ") axpy: " << modified_time_ms << " ms, ";
        std::cout << (bytes / modified_time_ms) * 1.0e-6 << " GB/s" << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);

}
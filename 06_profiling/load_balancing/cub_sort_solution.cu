#include "random.hpp"
#include "timer.hpp"

#include <cub/cub.cuh>

#include <iostream>

using key_t = int;
using value_t = double;

static constexpr int block = 256;

__global__ void unbalanced(double * out, double * in, int * iterations) {

    __shared__ double shmem[block];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int iter = iterations[tid];
    double value = in[tid];

    for (int i = 0; i < iter; i++) {
        value = sin(value);
    }

    shmem[threadIdx.x] = value;

    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){ 
        out[blockIdx.x] = shmem[0]; 
    }

}

__global__ void sort_first(double * out, double * in, int * iterations) {

    __shared__ double shmem[block];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Specialize BlockRadixSort for a 1D block owning 1 keys / values pair per thread
    using BlockRadixSort = cub::BlockRadixSort<key_t, block, 1, value_t>;

    // Allocate shared memory for BlockRadixSort
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    int iter[1] = {iterations[tid]};
    double value[1] = {in[tid]};

    BlockRadixSort(temp_storage).Sort(iter, value);

    for (int i = 0; i < iter[0]; i++) {
        value[0] = sin(value[0]);
    }

    // shared reduction

    shmem[threadIdx.x] = value[0];

    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){ 
        out[blockIdx.x] = shmem[0]; 
    }

    out[tid] = value[0];
}

int main() {

    int n = 1 << 22;

    int * d_iterations;
    double * d_in;
    double * d_out;

    cudaMalloc(&d_in, sizeof(double) * n);
    cudaMalloc(&d_out, sizeof(double) * n);
    cudaMalloc(&d_iterations, sizeof(int) * n);

    std::vector< int > h_iterations(n);
    std::vector< double > h_values(n);
    for (int i = 0; i < n; i++) {
        h_iterations[i] = random_integer(1, 256);
        h_values[i] = random_real(-1.0, 1.0);
    }

    cudaMemcpy(d_iterations, &h_iterations[0], sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, &h_values[0], sizeof(double) * n, cudaMemcpyHostToDevice);

    float unbalanced_time_ms = kernel_time_in_ms([&](){
        int grid = n / block;
        unbalanced<<< grid, block >>>(d_out, d_in, d_iterations);
    });

    std::cout << "unbalanced: " << unbalanced_time_ms << " ms" << std::endl;

    float sorted_time_ms = kernel_time_in_ms([&](){
        int grid = n / block;
        sort_first<<< grid, block >>>(d_out, d_in, d_iterations);
    });

    std::cout << "sort first: " << sorted_time_ms << " ms" << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_iterations);

}
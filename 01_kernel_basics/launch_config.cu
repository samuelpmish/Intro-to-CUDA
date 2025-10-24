#include <cstdio>

__global__ void kernel() {
    printf("hello world, from the GPU: thread %d, block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    printf("hello world, from the CPU\n");

    // launch our CUDA kernel with 4 block, and 4 threads / block
    int blocks_per_grid = 1;
    int threads_per_block = 1;
    kernel<<<blocks_per_grid, threads_per_block>>>();
    cudaDeviceSynchronize();
}
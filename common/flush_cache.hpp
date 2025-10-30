#pragma once

template < typename T >
__global__ void flush_cache_kernel(T * data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    data[tid]++;
}

void flush_cache() {
    int n = 1 << 24;
    double * d_data;
    cudaMalloc(&d_data, sizeof(double) * n);
    flush_cache_kernel<<< n / 256, 256 >>>(d_data);
    cudaFree(d_data);
}
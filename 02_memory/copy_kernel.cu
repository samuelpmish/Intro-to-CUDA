#include <iostream>
#include <cinttypes>

__global__ void copy_kernel(double * out, double * in, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = in[i] + 1;
    }
}

int main() {

    int num_runs = 5;
    int max_blocks = (1 << 20);
    int threads_per_block = 64;

    double * d_in;
    double * d_out;

    cudaMalloc(&d_in, threads_per_block * max_blocks * sizeof(double));
    cudaMalloc(&d_out, threads_per_block * max_blocks * sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    copy_kernel<<< 1, 1 >>>(d_out, d_in, 1);
    cudaDeviceSynchronize();

    std::cout << "number of blocks, bandwidth (GB/s)" << std::endl;
    for (int blocks = 1; blocks < max_blocks; blocks *= 2) {
        cudaEventRecord(start);
        for (int k = 0; k < num_runs; k++) {
            copy_kernel<<< blocks, threads_per_block >>>(d_out, d_in, blocks * threads_per_block);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);

        uint64_t num_bytes = 2 * blocks * threads_per_block * sizeof(double) * num_runs;
        std::cout << blocks << ", " << num_bytes * 1.0e-6f / time_ms << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_in);
    cudaFree(d_out);

}
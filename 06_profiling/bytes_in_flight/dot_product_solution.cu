#include "chunk.hpp"
#include "timer.hpp"

#include <vector>
#include <iostream>

#define WHICH 0

template < int k, typename scalar_t >
__global__ void dot(const scalar_t * u, const scalar_t * v, int n, double * sum){

    extern __shared__ double shmem[]; 

    int tid = threadIdx.x + blockIdx.x * blockDim.x; 

#if WHICH==0
    // grid stride loop
    double local_total = 0;
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
        local_total += u[i] * v[i];
    }
#endif

#if WHICH==1
    scalar_t u_tmp[k] = {};
    scalar_t v_tmp[k] = {};
    for (int i = 0; i < k; i++) {
        int j = tid + i * gridDim.x * blockDim.x;
        if (j < n) {
            u_tmp[i] = u[j];
            v_tmp[i] = v[j];
        }
    }

    double local_total = 0;
    for (int i = 0; i < k; i++) {
        local_total += u_tmp[i] * v_tmp[i];
    }
#endif

    shmem[threadIdx.x] = local_total;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){ 
        atomicAdd(sum, shmem[0]);
    }

}

template < int k, typename data_t >
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

    double * d_out;
    cudaMalloc(&d_out, sizeof(double));
    cudaMemset(d_out, 0, sizeof(double));

    int block = 512;
    int grid = ((n / k) + block - 1) / block;
    int shmem = block * sizeof(double);
    dot<k><<<grid, block, shmem>>>(d_a, d_b, 0, d_out);
    cudaDeviceSynchronize();

    stopwatch.start();
    dot<k><<<grid,block,shmem>>>(d_a, d_b, n, d_out);
    cudaDeviceSynchronize();
    stopwatch.stop();

    double h_out;
    cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost);

    uint64_t bytes = 2 * sizeof(data_t) * n;
    std::cout << h_out << ": " << stopwatch.elapsed() * 1000.0f << " ms ";
    std::cout << (bytes / stopwatch.elapsed()) * 1.0e-9 << " GB/s" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

}

int main() {

    std::cout << "fp32: "; run_test< 8, float >(1 << 24);
    std::cout << "fp64: "; run_test< 8, double >(1 << 24);
    std::cout << "  u8: "; run_test< 8, uint8_t >(1 << 24);
    std::cout << " u16: "; run_test< 8, uint16_t >(1 << 24);
    std::cout << " u32: "; run_test< 8, uint32_t >(1 << 24);
    std::cout << " u64: "; run_test< 8, uint64_t >(1 << 24);

}
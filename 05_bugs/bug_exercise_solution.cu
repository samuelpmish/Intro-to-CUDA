#include <iostream>

#include "timer.hpp"

//  1
// ⌠   4        
// |  ---  dx = π
// ⌡  1+x²      
//  0 
__global__ void calculate_pi(int n, double * sum){

    // shared storage used to communicate between threads
    extern __shared__ double shmem[]; 

    auto f = [](double x) { return 4.0 / (1.0 + x * x); };

    double dx = 1.0 / n;
    int i = threadIdx.x + blockIdx.x * blockDim.x; 
    if (i < n) {
        double x = (i + 0.5) * dx;
        shmem[threadIdx.x] = f(x) * dx;
    } else {
        shmem[threadIdx.x] = 0.0;
    }

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


int main() {

    static constexpr int n = 10'000'000;
    static constexpr double pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862090;

    // allocate memory for the answer on the GPU, and initialize it to zero
    double * d_pi_approx;
    cudaMalloc(&d_pi_approx, sizeof(double));
    cudaMemset(d_pi_approx, 0, sizeof(double));

    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    int shmem = sizeof(double) * threads_per_block;
    float time_ms = 1000.0f * time([&](){
        calculate_pi<<< blocks_per_grid, threads_per_block, shmem >>>(n, d_pi_approx);
        cudaDeviceSynchronize();
    });

    // copy the answer back from the GPU to compare
    double pi_approx;
    cudaMemcpy(&pi_approx, d_pi_approx, sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "computed pi ≈ " << pi_approx << " in " << time_ms << " ms, error " << fabs(pi - pi_approx) << std::endl;

    cudaFree(d_pi_approx);

}
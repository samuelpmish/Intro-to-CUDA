#include <iostream>

#include "timer.hpp"

//  1
// ⌠   4        
// |  ---  dx = π
// ⌡  1+x²      
//  0 
__global__ void calculate_pi(int n, double * block_sum){

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
        block_sum[blockIdx.x] = shmem[0];
    }

}

// note: this kernel should only be called with 1 block! 
__global__ void final_reduce(int n, double * block_sum, double * pi_approx) {

    // shared storage used to communicate between threads
    extern __shared__ double shmem[]; 

    double local_total = 0.0;
    // first accumulate entries in local_total
    // TODO

    // then write each thread's total to shmem
    // TODO

    // perform remaining reduction in shared memory as before
    // shmem = {a, b, c, d, e, f, g, h,  i, j, k, l, m, n, o, p}
    // shmem = {a+i, b+j, c+k, d+l, e+m, f+n, g+o, h+p,  ...}
    // shmem = {a+i+e+m, b+j+f+n, c+k+g+o, d+l+h+p, ... }
    // shmem = {a+i+e+m+c+k+g+o, b+j+f+n+d+l+h+p,  ... }
    // shmem = {a+i+e+m+c+k+g+o+b+j+f+n+d+l+h+p,  ... }

    // have one thread write out the result
    if (threadIdx.x == 0){
        *pi_approx = shmem[0];
    }

}

int main() {

    static constexpr int n = 100'000'000;
    static constexpr double pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862090;

    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // allocate memory for the answer on the GPU, and initialize it to zero
    double * d_pi_approx;
    cudaMalloc(&d_pi_approx, sizeof(double));

    // allocate memory for each block to write 1 value
    double * d_storage;
    cudaMalloc(&d_storage, blocks_per_grid * sizeof(double));

    cudaMemset(d_pi_approx, 0, sizeof(double));

    int shmem = sizeof(double) * threads_per_block;
    float time_ms = 1000.0f * time([&](){
        calculate_pi<<< blocks_per_grid, threads_per_block, shmem >>>(n, d_storage);
        final_reduce<<< 1, 1024, 1024 * sizeof(double) >>>(blocks_per_grid, d_storage, d_pi_approx);
        cudaDeviceSynchronize();
    });

    // copy the answer back from the GPU to compare
    double pi_approx;
    cudaMemcpy(&pi_approx, d_pi_approx, sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "computed pi ≈ " << pi_approx << " in " << time_ms << " ms, error " << fabs(pi - pi_approx) << std::endl;

    cudaFree(d_pi_approx);

}
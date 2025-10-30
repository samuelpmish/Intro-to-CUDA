#include "vector.hpp"

#include <cmath>
#include <vector>
#include <iostream>
#include <cinttypes>
#include <functional>

namespace gpu {

cudaStream_t default_stream = {0};
cudaMemPool_t mempool;
bool use_memory_pool = false;

void custom_alloc(void ** p, uint64_t nbytes) {
    if (use_memory_pool) {
        cudaMallocAsync(p, nbytes, default_stream);
    } else {
        cudaMalloc(p, nbytes);
    }
}

void custom_free(void * p) {
    if (use_memory_pool) {
        cudaFreeAsync(p, default_stream);
    } else {
        cudaFree(p);
    }
}

vector::vector(uint32_t n) : sz{}, ptr{} { resize(n); }

vector::vector(const vector & other) : sz{}, ptr{} {
  resize(other.sz);
  cudaMemcpy(ptr, other.ptr, sizeof(double) * other.sz, cudaMemcpyDeviceToDevice);
}

vector& vector::operator=(const vector & other) {
  resize(other.sz);
  cudaMemcpy(ptr, other.ptr, sizeof(double) * other.sz, cudaMemcpyDeviceToDevice);
  return *this;
}

vector::vector(const std::vector<double> & other) : sz{}, ptr{} {
  resize(other.size());
  cudaMemcpy(ptr, &other[0], sizeof(double) * other.size(), cudaMemcpyHostToDevice);
}

vector& vector::operator=(const std::vector<double> & other) {
  resize(other.size());
  cudaMemcpy(ptr, &other[0], sizeof(double) * other.size(), cudaMemcpyHostToDevice);
  return *this;
}

void vector::resize(uint32_t new_sz) {
  if (sz != new_sz) {
    if (ptr) { custom_free(ptr); }
    if (new_sz > 0) { custom_alloc((void**)&ptr, sizeof(double) * new_sz); }
    sz = new_sz;
  }
}

uint32_t vector::size() const { return sz; }

void vector::set_memory_pool(uint64_t bytes) {
    if (bytes > 0) {
        use_memory_pool = true;
        cudaDeviceGetDefaultMemPool(&mempool, 0);
        cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &bytes);
    } else {
        use_memory_pool = false;
    }
}

vector::~vector() { resize(0); }

vector zeros(int n) {
    vector out(n);
    cudaMemset(out.ptr, 0, sizeof(double) * n);
    return out;
}

__global__ void add(double * sum, double * a, double * b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) { sum[i] = a[i] + b[i]; }
}

vector operator+(const vector & u, const vector & v) {
    vector out(v.sz);
    int threads_per_block = 256;
    int blocks = (u.sz + threads_per_block - 1) / threads_per_block;
    add<<< blocks, threads_per_block >>>(out.ptr, u.ptr, v.ptr, u.sz);
    return out;
}

__global__ void sub(double * diff, double * a, double * b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) { diff[i] = a[i] - b[i]; }
}

vector operator-(const vector & u, const vector & v) {
    vector out(v.sz);
    int threads_per_block = 256;
    int blocks = (u.sz + threads_per_block - 1) / threads_per_block;
    sub<<< blocks, threads_per_block >>>(out.ptr, u.ptr, v.ptr, u.sz);
    return out;
}

__global__ void scale(double * out, const double * in, double scale, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) { out[i] = scale * in[i]; }
}

vector operator*(const vector & v, double s) { 
    vector out(v.sz);
    int threads_per_block = 256;
    int blocks = (v.sz + threads_per_block - 1) / threads_per_block;
    scale<<< blocks, threads_per_block >>>(out.ptr, v.ptr, s, v.sz);
    return out;
}

vector operator*(double s, const vector & v) {
    vector out(v.sz);
    int threads_per_block = 256;
    int blocks = (v.sz + threads_per_block - 1) / threads_per_block;
    scale<<< blocks, threads_per_block >>>(out.ptr, v.ptr, s, v.sz);
    return out;
}

vector operator/(const vector & v, double s) {
    vector out(v.sz);
    int threads_per_block = 256;
    int blocks = (v.sz + threads_per_block - 1) / threads_per_block;
    scale<<< blocks, threads_per_block >>>(out.ptr, v.ptr, 1.0 / s, v.sz);
    return out;
}

__global__ void axpby_kernel(double a, const double * x, double b, double * y, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) { y[i] = a * x[i] + b * y[i]; }
}

void axpby(double a, const vector & x, double b, vector & y) {
    int threads_per_block = 256;
    int blocks = (y.sz + threads_per_block - 1) / threads_per_block;
    axpby_kernel<<< blocks, threads_per_block >>>(a, x.ptr, b, y.ptr, y.sz);
    cudaDeviceSynchronize();
}

__global__ void dot_1(const double * u, const double * v, int n, double * block_sum){

    using real_t = float;

    extern __shared__ real_t shmem[]; 

    int i = threadIdx.x + blockIdx.x * blockDim.x; 
    if (i < n) {
        shmem[threadIdx.x] = u[i] * v[i];
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
__global__ void dot_2(int n, double * block_sum, double * sum) {

    // shared storage used to communicate between threads
    extern __shared__ double shmem[]; 

    // first accumulate entries in registers
    double thr_sum = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        thr_sum += block_sum[i];
    }

    // then write each thread's total to shmem
    shmem[threadIdx.x] = thr_sum;
    __syncthreads();

    // perform remaining reduction in shared memory as before
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // have one thread write out the result
    if (threadIdx.x == 0){
        *sum = shmem[0];
    }

}

double dot(const vector & u, const vector & v) { 
    int threads_per_block = 512;
    int blocks = (u.sz + threads_per_block - 1) / threads_per_block;
    int shmem = sizeof(double) * threads_per_block;

    double * d_sum;
    custom_alloc((void**)&d_sum, sizeof(double));

    double * d_block_sums;
    custom_alloc((void**)&d_block_sums, sizeof(double) * blocks);

    dot_1<<< blocks, threads_per_block, shmem >>>(u.ptr, v.ptr, u.sz, d_block_sums);
    dot_2<<< 1, threads_per_block, shmem >>>(blocks, d_block_sums, d_sum);

    double h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

    custom_free(d_block_sums);
    custom_free(d_sum);

    return h_sum;
}

} // namespace gpu
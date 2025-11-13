#include <cstdio>
#include <cinttypes>

#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

static constexpr int kmax = 100000;

template < typename callable, typename T >
__global__ void kernel_template(callable f, T * data) {

    volatile T local = *data;
    volatile T copy = local;

    for (int k = 0; k < kmax; k++) {
        local = f(local);
    }

    if (threadIdx.x > 512) {
        data[0] = local; // never executes
    }
}

template < typename callable, typename T >
float runtime_ms(callable f, T * ptr) {

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    kernel_template<<<1,1>>>(f, ptr);
    cudaEventRecord(start);
    kernel_template<<<1,1>>>(f, ptr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_ms;

}

int main() {

    float * d_f32;
    double * d_f64;
    int32_t * d_i32;
    int64_t * d_i64;
    uint32_t * d_u32;
    uint64_t * d_u64;

    cudaMalloc(&d_f32, sizeof(float) * 32);
    cudaMalloc(&d_f64, sizeof(double) * 32);
    cudaMalloc(&d_i32, sizeof(int32_t) * 32);
    cudaMalloc(&d_i64, sizeof(int64_t) * 32);
    cudaMalloc(&d_u32, sizeof(uint32_t) * 32);
    cudaMalloc(&d_u64, sizeof(uint64_t) * 32);

////////////////////////////////////////////////////////////////////////////////

    // fp32
    std::vector< float > h_f32(32, 0.1f);
    cudaMemcpy(d_f32, &h_f32[0], 32 * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "  no-op: " << runtime_ms(
        [] __device__ (float x){ return 0.0f; }, 
        d_f32
    ) << " ms" << std::endl << std::endl;

    std::cout << "f32: " << std::endl;

    std::cout << std::setw(20) << "  x + f32: " << runtime_ms(
        [] __device__ (float x){ return x + 0.1f; }, 
        d_f32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x * f32: " << runtime_ms(
        [] __device__ (float x){ return x * 2.1f; }, 
        d_f32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x / f32: " << runtime_ms(
        [] __device__ (float x){ return x / 0.8f; }, 
        d_f32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  f32 / x: " << runtime_ms(
        [] __device__ (float x){ return 42.0f / x; }, 
        d_f32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  f32 * x + f32: " << runtime_ms(
        [] __device__ (float x){ return 1.2f * x - 4.0f; }, 
        d_f32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  sin(x): " << runtime_ms(
        [] __device__ (float x){ return sinf(x); }, 
        d_f32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  tan(x): " << runtime_ms(
        [] __device__ (float x){ return tanf(x); }, 
        d_f32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  exp(x): " << runtime_ms(
        [] __device__ (float x){ return expf(x); }, 
        d_f32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  log(x): " << runtime_ms(
        [] __device__ (float x){ return logf(x); }, 
        d_f32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  asin(x): " << runtime_ms(
        [] __device__ (float x){ return asinf(x); }, 
        d_f32
    ) << " ms" << std::endl;

////////////////////////////////////////////////////////////////////////////////

    // fp64
    std::vector< double > h_f64(32, 10.0);
    cudaMemcpy(d_f64, &h_f64[0], 32 * sizeof(double), cudaMemcpyHostToDevice);

    std::cout << "f64: " << std::endl;

    std::cout << std::setw(20) << "  x + f64: " << runtime_ms(
        [] __device__ (double x){ return x + 3.0; }, 
        d_f64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x * f64: " << runtime_ms(
        [] __device__ (double x){ return x * 0.91; }, 
        d_f64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x / f64: " << runtime_ms(
        [] __device__ (double x){ return x / 42.0; }, 
        d_f64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  f64 / x: " << runtime_ms(
        [] __device__ (double x){ return 42.0 / x; }, 
        d_f64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  f64 * x + f64: " << runtime_ms(
        [] __device__ (double x){ return 1.2 * x - 4.0; }, 
        d_f64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  sin(x): " << runtime_ms(
        [] __device__ (double x){ return sin(x); }, 
        d_f64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  tan(x): " << runtime_ms(
        [] __device__ (double x){ return tan(x); }, 
        d_f64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  exp(x): " << runtime_ms(
        [] __device__ (double x){ return exp(x); }, 
        d_f64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  log(x): " << runtime_ms(
        [] __device__ (double x){ return log(x); }, 
        d_f64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  asin(x): " << runtime_ms(
        [] __device__ (double x){ return asin(x); }, 
        d_f64
    ) << " ms" << std::endl;

////////////////////////////////////////////////////////////////////////////////

    // i32
    std::vector< int32_t > h_i32(32, 10);
    cudaMemcpy(d_i32, &h_i32[0], 32 * sizeof(int32_t), cudaMemcpyHostToDevice);

    std::cout << "i32: " << std::endl;

    std::cout << std::setw(20) << "  x + i32: " << runtime_ms(
        [] __device__ (int32_t x){ return x + int32_t(3); }, 
        d_i32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x * i32: " << runtime_ms(
        [] __device__ (int32_t x){ return x * x; }, 
        d_i32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  i32 % x: " << runtime_ms(
        [] __device__ (int32_t x){ return int32_t(42) % x; }, 
        d_i32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x % i32: " << runtime_ms(
        [] __device__ (int32_t x){ return x % int32_t(67); }, 
        d_i32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  i32 / x: " << runtime_ms(
        [] __device__ (int32_t x){ return int32_t(42) / x; }, 
        d_i32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x / i32: " << runtime_ms(
        [] __device__ (int32_t x){ return x / int32_t(1362); }, 
        d_i32
    ) << " ms" << std::endl;

////////////////////////////////////////////////////////////////////////////////

    // i64
    std::vector< int64_t > h_i64(32, 10);
    cudaMemcpy(d_i64, &h_i64[0], 32 * sizeof(int64_t), cudaMemcpyHostToDevice);

    std::cout << "i64: " << std::endl;

    std::cout << std::setw(20) << "  x + i64: " << runtime_ms(
        [] __device__ (int64_t x){ return x + int64_t(3); }, 
        d_i64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x * i64: " << runtime_ms(
        [] __device__ (int64_t x){ return x * x; }, 
        d_i64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  i64 % x: " << runtime_ms(
        [] __device__ (int64_t x){ return int64_t(42) % x; }, 
        d_i64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x % i64: " << runtime_ms(
        [] __device__ (int64_t x){ return x % int64_t(67); }, 
        d_i64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  i64 / x: " << runtime_ms(
        [] __device__ (int64_t x){ return int64_t(42) / x; }, 
        d_i64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x / i64: " << runtime_ms(
        [] __device__ (int64_t x){ return x / int64_t(1362); }, 
        d_i64
    ) << " ms" << std::endl;

////////////////////////////////////////////////////////////////////////////////

    // u32
    std::vector< uint32_t > h_u32(32, 10);
    cudaMemcpy(d_u32, &h_u32[0], 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    std::cout << "u32: " << std::endl;

    std::cout << std::setw(20) << "  x + u32: " << runtime_ms(
        [] __device__ (uint32_t x){ return x + uint32_t(3); }, 
        d_u32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x * u32: " << runtime_ms(
        [] __device__ (uint32_t x){ return x * x; }, 
        d_u32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  u32 % x: " << runtime_ms(
        [] __device__ (uint32_t x){ return uint32_t(42) % x; }, 
        d_u32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x % u32: " << runtime_ms(
        [] __device__ (uint32_t x){ return x % uint32_t(67); }, 
        d_u32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  u32 / x: " << runtime_ms(
        [] __device__ (uint32_t x){ return uint32_t(42) / x; }, 
        d_u32
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x / u32: " << runtime_ms(
        [] __device__ (uint32_t x){ return x / uint32_t(1362); }, 
        d_u32
    ) << " ms" << std::endl;

////////////////////////////////////////////////////////////////////////////////

    // u64
    std::vector< uint64_t > h_u64(32, 10);
    cudaMemcpy(d_u64, &h_u64[0], 32 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    std::cout << "u64: " << std::endl;

    std::cout << std::setw(20) << "  x + u64: " << runtime_ms(
        [] __device__ (uint64_t x){ return x + uint64_t(3); }, 
        d_u64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x * u64: " << runtime_ms(
        [] __device__ (uint64_t x){ return x * x; }, 
        d_u64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  u64 % x: " << runtime_ms(
        [] __device__ (uint64_t x){ return uint64_t(42) % x; }, 
        d_u64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x % u64: " << runtime_ms(
        [] __device__ (uint64_t x){ return x % uint64_t(67); }, 
        d_u64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  u64 / x: " << runtime_ms(
        [] __device__ (uint64_t x){ return uint64_t(42) / x; }, 
        d_u64
    ) << " ms" << std::endl;

    std::cout << std::setw(20) << "  x / u64: " << runtime_ms(
        [] __device__ (uint64_t x){ return x / uint64_t(1362); }, 
        d_u64
    ) << " ms" << std::endl;

////////////////////////////////////////////////////////////////////////////////

    cudaFree(d_f32);
    cudaFree(d_f64);
    cudaFree(d_i32);
    cudaFree(d_i64);
    cudaFree(d_u32);
    cudaFree(d_u64);

}
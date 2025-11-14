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

    for (int k = 0; k < 100000; k++) { 
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

    double * d_f64;
    cudaMalloc(&d_f64, sizeof(double) * 32);

////////////////////////////////////////////////////////////////////////////////

    // fp32
    std::vector< double > h_f64(32, 0.9f);
    cudaMemcpy(d_f64, &h_f64[0], 32 * sizeof(double), cudaMemcpyHostToDevice);

    std::cout << std::setw(32) << "f(x): " << runtime_ms(
        [] __device__ (double x){ return sin(sin(sin(x))); }, 
        d_f64
    ) << " ms" << std::endl;

    std::cout << std::setw(32) << "Taylor approximant (2): " << runtime_ms(
        [] __device__ (double x){ return (x - 0.5*x*x*x); }, 
        d_f64
    ) << " ms" << std::endl;

    std::cout << std::setw(32) << "Taylor approximant (3): " << runtime_ms(
        [] __device__ (double x){ return (x - 0.5*x*x*x + 0.275*x*x*x*x*x); }, 
        d_f64
    ) << " ms" << std::endl;

    std::cout << std::setw(32) << "Taylor approximant (3H): " << runtime_ms(
        [] __device__ (double x){ return x*(1 + x*x*(-0.5 + x*x*0.275)); }, 
        d_f64
    ) << " ms" << std::endl;

    std::cout << std::setw(32) << "Pade approximant (2): " << runtime_ms(
        [] __device__ (double x){ return (x + 0.05*x*x*x)/(1.0 + 0.55*x*x); }, 
        d_f64
    ) << " ms" << std::endl;

    std::cout << std::setw(32) << "Pade approximant (3): " << runtime_ms(
        [] __device__ (double x){ return (x + 0.871406*x*x*x + 0.0534909*x*x*x*x*x)/(1.0 + 1.37141*x*x + 0.464194*x*x*x*x); }, 
        d_f64
    ) << " ms" << std::endl;

    std::cout << std::setw(32) << "Pade approximant (3H): " << runtime_ms(
        [] __device__ (double x){ 
            double x2 = x*x;
            return x*(1 + x2*(0.871406 + x2*0.0534909))/(1.0 + x2*(1.37141 + x2*0.464194)); }, 
        d_f64
    ) << " ms" << std::endl;

    cudaFree(d_f64);

}
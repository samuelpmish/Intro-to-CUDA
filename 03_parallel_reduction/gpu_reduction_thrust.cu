#include <iostream>

#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include "timer.hpp"

struct integrand {
  integrand(int n) : dx(1.0 / n) {}

  __device__ double f(double x) const {
      return 4.0 / (1.0 + x * x);
  }

  //  1
  // ⌠   4        
  // |  ---  dx = π
  // ⌡  1+x²      
  //  0 
  __device__ double operator()(const int & i) const {
    return f((i + 0.5) * dx) * dx;
  }

  double dx;
};

int main() {

    static constexpr int n = 100'000'000;
    static constexpr double pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862090;

    double pi_approx;

    thrust::counting_iterator<int> iota(0);

    // do any CUDA runtime initialization outside of the timed block of code
    cudaDeviceSynchronize();

    float time_ms = 1000.0f * time([&](){
        pi_approx = thrust::transform_reduce(thrust::device,
                                             iota, iota + n,
                                             integrand(n),
                                             0.0,
                                             thrust::plus<double>());
    });

    std::cout << "computed pi ≈ " << pi_approx << " in " << time_ms << " ms, error " << fabs(pi - pi_approx) << std::endl;

}
#include "span.hpp"
#include "timer.hpp"

#include <vector>
#include <iostream>

template < typename T >
__global__ void laplace_operator(span3D<T> output, const span3D<T> input) {

  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  int iz = threadIdx.z + blockIdx.z * blockDim.z;

  bool interior = (0 < ix) && (ix < input.shape_[2] - 1) &&
                  (0 < iy) && (iy < input.shape_[1] - 1) &&
                  (0 < iz) && (iz < input.shape_[0] - 1);

  if (interior) {
    output(iz, iy, ix) = input(iz+1, iy  , ix  )
                       + input(iz-1, iy  , ix  )
                       + input(iz  , iy+1, ix  )
                       + input(iz  , iy-1, ix  )
                       + input(iz  , iy  , ix+1)
                       + input(iz  , iy  , ix-1) 
                       - input(iz  , iy  , ix  ) * T{6.0};
  } else {
    output(iz, iy, ix) = input(iz, iy, ix);
  }

}

int main() {

  int n = 256;
  int shape[3] = {n, n, n};
  int num_runs = 3;

  double * d_input;
  double * d_output;

  cudaMalloc(&d_input, sizeof(double) * n * n * n);
  cudaMalloc(&d_output, sizeof(double) * n * n * n);
  cudaMemset(d_input, 0, n * n * n * sizeof(double));

  timer stopwatch;

  // let's stick to dimensions that evenly divide n to keep it simple
  std::vector< dim3 > blocks = {
    dim3{8,8,8},
    dim3{4,8,16},
    dim3{16,8,4},
    dim3{32,8,2},
    dim3{2,8,32},
    dim3{64,8,1},
    dim3{1,8,64},
    dim3{128,4,1},
    dim3{1,4,128}
  };

  for (dim3 block : blocks) {
    stopwatch.start();
    dim3 grid{n / block.x, n / block.y, n / block.z};
    for (int k = 0; k < num_runs; k++) {
      laplace_operator<<< grid, block >>>(
        span3D<double>{d_input, shape},
        span3D<double>{d_output, shape}
      );
    }
    cudaDeviceSynchronize();
    stopwatch.stop();

    std::cout << block.x << "x" << block.y << "x" << block.z << ": " << stopwatch.elapsed() * 1000.0f / num_runs << " ms" << std::endl;
  }

}
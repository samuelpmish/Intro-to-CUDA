#include "span.hpp"

#include "timer.hpp"

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

  double * d_input;
  double * d_output;

  cudaMalloc(&d_input, sizeof(double) * n * n * n);
  cudaMalloc(&d_output, sizeof(double) * n * n * n);
  cudaMemset(d_input, 0, n * n * n * sizeof(double));

  timer stopwatch;

  stopwatch.start();
  // assumes n is divisible by block dimensions, for simplicity
  dim3 block{8, 8, 8};
  dim3 grid{n / block.x, n / block.y, n / block.z};
  laplace_operator<<< grid, block >>>(
    span3D<double>{d_input, shape},
    span3D<double>{d_output, shape}
  );
  cudaDeviceSynchronize();
  stopwatch.stop();

  std::cout << block.x << "x" << block.y << "x" << block.z << ": " << stopwatch.elapsed() * 1000.0f << " ms" << std::endl;

}
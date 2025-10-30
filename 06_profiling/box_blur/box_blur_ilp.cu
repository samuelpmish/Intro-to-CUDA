#include "span.hpp"

#include "timer.hpp"

#include <iostream>

template < typename T >
__global__ void box_blur(span3D<T> output, const span3D<T> input) {

  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  int iz = threadIdx.z + blockIdx.z * blockDim.z;

  bool interior = (0 < ix) && (ix < input.shape_[2] - 1) &&
                  (0 < iy) && (iy < input.shape_[1] - 1) &&
                  (0 < iz) && (iz < input.shape_[0] - 1);

  if (interior) {
    T sum = {};
    for (int dz = -1; dz <= 1; dz++) {
      int z = iz + dz;
      for (int dy = -1; dy <= 1; dy++) {
        int y = iy + dy;
        for (int dx = -1; dx <= 1; dx++) {
          int x = ix + dx;
          sum += input(z, y, x);
        }
      }
    }
    output(iz, iy, ix) = sum / T{9};
  } else {
    output(iz, iy, ix) = input(iz, iy, ix);
  }

}

int main() {

  using real_t = float;

  int n = 512;
  int shape[3] = {n, n, n};

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  real_t * d_in;
  real_t * d_out;
  cudaMalloc(&d_in, sizeof(real_t) * n * n * n);
  cudaMalloc(&d_out, sizeof(real_t) * n * n * n);
  
  // assumes n is divisible by block dimensions, for simplicity
  dim3 block{8, 8, 8};
  dim3 grid{n / block.x, n / block.y, n / block.z};

  // skip timing the first kernel launch
  box_blur<<< 1, 1 >>>(span3D<real_t>{d_out, shape}, span3D<real_t>{d_in, shape});

  cudaEventRecord(start);
  box_blur<<< grid, block >>>(span3D<real_t>{d_out, shape}, span3D<real_t>{d_in, shape});
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float time_ms;
  cudaEventElapsedTime(&time_ms, start, stop);

  std::cout << "blur finished in " << time_ms << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

}
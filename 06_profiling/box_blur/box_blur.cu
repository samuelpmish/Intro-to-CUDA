#include "span.hpp"

#include "timer.hpp"

#include <vector>
#include <iostream>

template < typename T >
__global__ void compare(span3D<T> output, const span3D<T> input) {

  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  int iz = threadIdx.z + blockIdx.z * blockDim.z;

  bool interior = (0 < ix) && (ix < input.shape_[2] - 1) &&
                  (0 < iy) && (iy < input.shape_[1] - 1) &&
                  (0 < iz) && (iz < input.shape_[0] - 1);

  if (interior) {
    T error = input(iz, iy, ix) - output(iz, iy, ix);
    if (error * error > 1.0e-10) {
      printf("%d, %d, %d, %f %f\n", ix, iy, iz, input(iz, iy, ix), output(iz, iy, ix));
    }
  }
}

template < typename T >
__global__ void box_blur_2x2x2(span3D<T> output, const span3D<T> input) {

  int ix = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
  int iy = 2 * (threadIdx.y + blockIdx.y * blockDim.y);
  int iz = 2 * (threadIdx.z + blockIdx.z * blockDim.z);

  T scale = T{1} / T{27};

  auto close = [](int a, int b) { return (b - a) * (b - a) <= 1; };

  T sum[2][2][2] = {};
  int count = 0;
  for (int dz = -1; dz <= 2; dz++) {
    int z = iz + dz;
    if (z < 0 || z > input.shape_[0] - 1) { continue; }

    for (int dy = -1; dy <= 2; dy++) {
      int y = iy + dy;
      if (y < 0 || y > input.shape_[1] - 1) { continue; }

      for (int dx = -1; dx <= 2; dx++) {
        int x = ix + dx;
        if (x < 0 || x > input.shape_[2] - 1) { continue; }

        count++;
        T value = input(z, y, x);

        if (close(dz, 0) && close(dy, 0) && close(dx, 0)) sum[0][0][0] += value;
        if (close(dz, 0) && close(dy, 0) && close(dx, 1)) sum[0][0][1] += value;
        if (close(dz, 0) && close(dy, 1) && close(dx, 0)) sum[0][1][0] += value;
        if (close(dz, 0) && close(dy, 1) && close(dx, 1)) sum[0][1][1] += value;
        if (close(dz, 1) && close(dy, 0) && close(dx, 0)) sum[1][0][0] += value;
        if (close(dz, 1) && close(dy, 0) && close(dx, 1)) sum[1][0][1] += value;
        if (close(dz, 1) && close(dy, 1) && close(dx, 0)) sum[1][1][0] += value;
        if (close(dz, 1) && close(dy, 1) && close(dx, 1)) sum[1][1][1] += value;
      }
    }
  }

  for (int dz = 0; dz < 2; dz++) {
    int z = iz + dz;
    if (z == 0 || z == input.shape_[0] - 1) { continue; }
    for (int dy = 0; dy < 2; dy++) {
      int y = iy + dy;
      if (y == 0 || y == input.shape_[1] - 1) { continue; }
      for (int dx = 0; dx < 2; dx++) {
        int x = ix + dx;
        if (x == 0 || x == input.shape_[2] - 1) { continue; }
        output(z, y, x) = sum[dz][dy][dx] * scale;
      }
    }
  }

}

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
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          sum += input(iz+dz, iy+dy, ix+dx);
        }
      }
    }
    output(iz, iy, ix) = sum / T{27};
  }

}

int main() {

  using real_t = double;

  int n = 256;
  int shape[3] = {n, n, n};
  float time_ms;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  real_t * d_in;
  real_t * d_out;
  real_t * d_expected;
  cudaMalloc(&d_in, sizeof(real_t) * n * n * n);
  cudaMalloc(&d_out, sizeof(real_t) * n * n * n);
  cudaMalloc(&d_expected, sizeof(real_t) * n * n * n);

  std::vector< real_t > h_in(n * n * n, real_t{1});
  cudaMemcpy(d_in, &h_in[0], sizeof(real_t) * n * n * n, cudaMemcpyHostToDevice);

  cudaMemset(d_out, 0, sizeof(real_t) * n * n * n);
  cudaMemset(d_expected, 0, sizeof(real_t) * n * n * n);
  
  // assumes n is divisible by block dimensions, for simplicity
  dim3 block{8, 8, 8};
  dim3 grid{n / block.x, n / block.y, n / block.z};

  // skip timing the first kernel launch
  box_blur<<< 1, 1 >>>(span3D<real_t>{d_out, shape}, span3D<real_t>{d_in, shape});

  cudaEventRecord(start);
  box_blur<<< grid, block >>>(span3D<real_t>{d_expected, shape}, span3D<real_t>{d_in, shape});
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms, start, stop);
  std::cout << "original box blur finished in " << time_ms << " ms" << std::endl;

  box_blur_2x2x2<<< 1, 1 >>>(span3D<real_t>{d_out, shape}, span3D<real_t>{d_in, shape});
  cudaDeviceSynchronize();

  dim3 grid2{n / (2 * block.x), n / (2 * block.y), n / (2 * block.z)};
  cudaEventRecord(start);
  box_blur_2x2x2<<< grid2, block >>>(span3D<real_t>{d_out, shape}, span3D<real_t>{d_in, shape});
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms, start, stop);
  std::cout << "2x2x2 box blur finished in " << time_ms << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  compare<<< grid, block >>>(span3D<real_t>{d_out, shape}, span3D<real_t>{d_expected, shape});
  cudaDeviceSynchronize();

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_expected);

}
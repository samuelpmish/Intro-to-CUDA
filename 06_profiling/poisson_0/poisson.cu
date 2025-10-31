#include "vector.hpp"
#include "timer.hpp"
#include "span.hpp"

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

double residual_norm(const gpu::vector & x, const gpu::vector & b, int n) {
  int shape[3] = {n, n, n};

  gpu::vector Ax(x.size()); 

  span3D<double> Ax_3D(Ax.ptr, shape);
  span3D<double> x_3D(x.ptr, shape);

  // assumes n is divisible by block dimensions, for simplicity
  dim3 block{8, 8, 8};
  dim3 grid{n / block.x, n / block.y, n / block.z};
  laplace_operator<<< grid, block >>>(Ax_3D, x_3D);

  gpu::vector r = Ax - b;

  return sqrt(dot(r, r));
}

template < typename T >
__global__ void relaxation_operator(span3D<T> output, const span3D<T> input, const span3D<T> b) {

  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  int iz = threadIdx.z + blockIdx.z * blockDim.z;

  bool interior = (0 < ix) && (ix < input.shape_[2] - 1) &&
                  (0 < iy) && (iy < input.shape_[1] - 1) &&
                  (0 < iz) && (iz < input.shape_[0] - 1);

  if (interior) {
    output(iz, iy, ix) = (input(iz+1, iy  , ix  )
                        + input(iz-1, iy  , ix  )
                        + input(iz  , iy+1, ix  )
                        + input(iz  , iy-1, ix  )
                        + input(iz  , iy  , ix+1)
                        + input(iz  , iy  , ix-1) 
                        -     b(iz  , iy  , ix  )) / T{6.0};
  } else {
    output(iz, iy, ix) = input(iz, iy, ix);
  }

}

gpu::vector relaxation(const gpu::vector & x, const gpu::vector & b, int n) {
  int shape[3] = {n, n, n};

  gpu::vector x_next(x.sz);

  span3D<double> x_next_3D(x_next.ptr, shape);
  span3D<double> x_3D(x.ptr, shape);
  span3D<double> b_3D(b.ptr, shape);

  // assumes n is divisible by block dimensions, for simplicity
  dim3 block{8, 8, 8};
  dim3 grid{n / block.x, n / block.y, n / block.z};
  relaxation_operator<<< grid, block >>>(x_next_3D, x_3D, b_3D);

  return x_next;
}

int main() {

  int n = 256;
  int max_iterations = 200;
  double tolerance = 1.0e-2;

  // solution = 0 on the boundary, and we 
  // have a Dirac delta source on the interior
  std::vector<double> rhs(n * n * n, 0.0);
  rhs[(n / 2) * n * n + (n / 2) * n + (n / 2)] = 1.0;
  gpu::vector b = rhs;

  timer stopwatch;

  // iterative solve by Jacobi method
  stopwatch.start();
  gpu::vector x(std::vector<double>(n * n * n, 0.0));
  for (int k = 0; k < max_iterations; k++) {
    x = relaxation(x, b, n);
    double norm_r = residual_norm(x, b, n);
    std::cout << k << " " << norm_r << std::endl;
    if (norm_r < tolerance) {
      break;
    }
  }
  cudaDeviceSynchronize();
  stopwatch.stop();

  std::cout << "finished in " << stopwatch.elapsed() * 1000.0f << " ms" << std::endl;

}
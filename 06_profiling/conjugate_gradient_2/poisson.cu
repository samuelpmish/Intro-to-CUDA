#include "krylov.hpp"

#include "span.hpp"
#include "vector.hpp"

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
  int max_iterations = 50;
  double tolerance = 1.0e-5;

  gpu::vector::set_memory_pool(n * n * n * sizeof(double) * 8);

  auto A = [&](const gpu::vector & x){
    gpu::vector Ax(x.size()); 

    span3D<double> Ax_3D(Ax.ptr, shape);
    span3D<double> x_3D(x.ptr, shape);

    // assumes n is divisible by block dimensions, for simplicity
    dim3 block{8, 8, 8};
    dim3 grid{n / block.x, n / block.y, n / block.z};
    laplace_operator<<< grid, block >>>(Ax_3D, x_3D);
    //cudaDeviceSynchronize();

    return Ax;
  };
  
  // solution = 0 on the boundary, and we 
  // have a Dirac delta source on the interior
  std::vector<double> rhs(n * n * n, 0.0);
  rhs[(n / 2) * n * n + (n / 2) * n + (n / 2)] = 1.0;
  gpu::vector b = rhs;

  timer stopwatch;

  stopwatch.start();
  gpu::vector x = cg(A, b, max_iterations, tolerance);
  cudaDeviceSynchronize();
  stopwatch.stop();

  std::cout << "finished in " << stopwatch.elapsed() * 1000.0f << " ms" << std::endl;

}
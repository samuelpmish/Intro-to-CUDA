#pragma once

template < typename T >
__global__ void laplace_scatter(T * out, const T * in, int nx, int ny, int nz){

  auto id = [nx, ny](int ix, int iy, int iz) {
    return ix + nx * (iy + ny * iz);
  };

  // only interior cells receive contributions, to match
  // the boundary treatment of the gather kernels
  auto interior = [nx, ny, nz](int ix, int iy, int iz) {
    return (0<ix) && (ix<(nx-1)) && (0<iy) && (iy<(ny-1)) && (0<iz) && (iz<(nz-1));
  };

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  if ((i < nx) && (j < ny) && (k < nz)) {
    T v = in[id(i,j,k)];

    if (interior(i  , j  , k-1)) atomicAdd(&out[id(i  , j  , k-1)], v);
    if (interior(i  , j-1, k  )) atomicAdd(&out[id(i  , j-1, k  )], v);
    if (interior(i-1, j  , k  )) atomicAdd(&out[id(i-1, j  , k  )], v);
    if (interior(i  , j  , k  )) atomicAdd(&out[id(i  , j  , k  )], v * T(-6.0));
    if (interior(i+1, j  , k  )) atomicAdd(&out[id(i+1, j  , k  )], v);
    if (interior(i  , j+1, k  )) atomicAdd(&out[id(i  , j+1, k  )], v);
    if (interior(i  , j  , k+1)) atomicAdd(&out[id(i  , j  , k+1)], v);
  }

}

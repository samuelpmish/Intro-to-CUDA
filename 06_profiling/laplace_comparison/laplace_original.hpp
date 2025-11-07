#pragma once

template < typename T >
__global__ void laplace_original(T * out, const T * in, int nx, int ny, int nz){

#ifndef ID_MACRO
  auto id = [nx, ny](int ix, int iy, int iz) {
    return ix + nx * (iy + ny * iz);
  };
#endif

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  // only update values on the interior
  if((0<i) && (i<(nx-1)) && (0<j) && (j<(ny-1)) && (0<k) && (k<(nz-1))) {
    out[id(i,j,k)] = in[id(i  , j  , k-1)]
                   + in[id(i  , j-1, k  )]
                   + in[id(i-1, j  , k  )]
                   - in[id(i  , j  , k  )] * (6.0)
                   + in[id(i+1, j  , k  )]
                   + in[id(i  , j+1, k  )]
                   + in[id(i  , j  , k+1)];
  }

}
#pragma once

#include "chunk.hpp"

template < int m, typename T >
__global__ void laplace_vectorized(T * out, const T * in, int nx, int ny, int nz) {

#ifndef ID_MACRO
  auto id = [nx, ny](int ix, int iy, int iz) {
    return ix + nx * (iy + ny * iz);
  };
#endif

  int i0 = m * (threadIdx.x + blockIdx.x * blockDim.x);
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  // only update values on the interior
  if((i0<=(nx-m)) && (0<j) && (j<(ny-1)) && (0<k) && (k<(nz-1))) {

    chunk<T,m> center_values = aligned_load_chunk<m>(&in[id(i0,j,k)]);

    chunk<T,m> tmp{};
    if (i0 > 0) { tmp[0] += in[id(i0-1,j,k)]; }
    for (int di = 0; di < m; di++) {
      tmp[di] -= 6 * center_values[di];
      if (di > 0) tmp[di] += center_values[di-1];
      if (di < m-1) tmp[di] += center_values[di+1];
    }
    if (i0+m < nx) {
      tmp[m-1] += in[id(i0+m,j,k)];
    }

    tmp += aligned_load_chunk<m>(&in[id(i0,j  ,k-1)]) 
         + aligned_load_chunk<m>(&in[id(i0,j-1,k  )]) 
         + aligned_load_chunk<m>(&in[id(i0,j+1,k  )]) 
         + aligned_load_chunk<m>(&in[id(i0,j  ,k+1)]);

    aligned_store_chunk(&out[id(i0,j,k)], tmp);

  }

}
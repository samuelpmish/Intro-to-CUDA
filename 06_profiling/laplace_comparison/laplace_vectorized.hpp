#pragma once

#include "chunk.hpp"

template < int m, typename T >
__global__ void laplace_vectorized(T * out, const T * in, int nx, int ny, int nz) {

  auto id = [nx, ny](int ix, int iy, int iz) {
    return ix + nx * (iy + ny * iz);
  };

  int i0 = m * (threadIdx.x + blockIdx.x * blockDim.x);
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  // only update values on the interior
  if((i0<=(nx-m)) && (0<j) && (j<(ny-1)) && (0<k) && (k<(nz-1))) {

    chunk<T,m> center_values = aligned_load_chunk<m>(&in[id(i0,j,k)]);

    // the two x-neighbors that fall outside this chunk
    T xm_edge = (i0   >  0) ? in[id(i0-1,j,k)] : T{};
    T xp_edge = (i0+m < nx) ? in[id(i0+m,j,k)] : T{};

    // accumulate the seven terms in the same order as laplace_original, so
    // that each lane rounds identically to the scalar kernel
    chunk<T,m> tmp = aligned_load_chunk<m>(&in[id(i0,j  ,k-1)]);
    tmp           += aligned_load_chunk<m>(&in[id(i0,j-1,k  )]);

    #pragma unroll
    for (int di = 0; di < m; di++) {
      tmp[di] += (di > 0)   ? center_values[di-1] : xm_edge;
      tmp[di] -= center_values[di] * T(6.0);
      tmp[di] += (di < m-1) ? center_values[di+1] : xp_edge;
    }

    tmp += aligned_load_chunk<m>(&in[id(i0,j+1,k  )]);
    tmp += aligned_load_chunk<m>(&in[id(i0,j  ,k+1)]);

    // laplace_original does not touch the x boundary, so this kernel must not
    // change it either.  Rather than branching to a scalar store (which splits
    // the warp holding the edge chunk, and costs ~6%), put the value that is
    // already there back into the edge lanes and keep one wide store on every
    // path.  Each output cell has exactly one writer, so this is a no-op.
    if (i0   == 0)  { tmp[0]   = out[id(0,    j, k)]; }
    if (i0+m >= nx) { tmp[m-1] = out[id(nx-1, j, k)]; }

    aligned_store_chunk(&out[id(i0,j,k)], tmp);

  }

}

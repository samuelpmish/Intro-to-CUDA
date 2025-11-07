#pragma once

#include "chunk.hpp"

template < int mx, int my, int mz, typename T >
__global__ void laplace_tiled_3D(T * out, const T * in, int nx, int ny, int nz) {

  constexpr int zstages = (mz >= 2) ? 2 : 1;

  auto id = [nx, ny](int ix, int iy, int iz) {
    return ix + nx * (iy + ny * iz);
  };

  int i0 = mx * (threadIdx.x + blockIdx.x * blockDim.x);
  int j0 = my * (threadIdx.y + blockIdx.y * blockDim.y);
  int k0 = mz * (threadIdx.z + blockIdx.z * blockDim.z);

  // only update values on the interior
  if((i0<(nx-mx+1)) && (j0<(ny-my+1)) && ((1-mz)<k0) && (k0<(nz-mz+1))) {

    chunk<T,mx> tmp[zstages][my] = {};

    #pragma unroll
    for (int dk = -1; dk < mz + 1; dk++) {

      int k = k0 + dk;
      if(k < 0 || nz <= k) continue;

      int dkm2 = (dk+2) % 2;

      #pragma unroll
      for (int dj = -1; dj < my + 1; dj++) {

        int j = j0 + dj;
        if(j < 0 || ny <= j) continue;

        chunk<T,mx> c = aligned_load_chunk<mx>(&in[id(i0,j,k)]);

        #pragma unroll
        for (int di = 0; di < mx; di++) {

          // center contribution
          if (0 <= dk && dk < mz && 0 <= dj && dj < my) {
            tmp[dkm2][dj][di] -= T(6) * c[di];
          }

          // +x contribution
          if (0 <= dk && dk < mz && 0 <= dj && dj < my && di+1 < mx) {
            tmp[dkm2][dj][di+1] += c[di];
          }

          // -x contribution
          if (0 <= dk && dk < mz && 0 <= dj && dj < my && 0 <= di-1) {
            tmp[dkm2][dj][di-1] += c[di];
          }

          // +x contribution (edge)
          if (0 <= dk && dk < mz && 0 <= dj && dj < my && 0 <= i0-1 && di == 0) {
            tmp[dkm2][dj][di] += in[id(i0-1,j,k)];
          }

          // -x contribution (edge)
          if (0 <= dk && dk < mz && 0 <= dj && dj < my && i0+mx < nx && di == mx-1) {
            tmp[dkm2][dj][di] += in[id(i0+mx,j,k)];
          }

          // +y contribution
          if (0 <= dk && dk < mz && dj+1 < my) {
            tmp[dkm2][dj+1][di] += c[di];
          }

          // -y contribution
          if (0 <= dk && dk < mz && 0 <= dj-1) {
            tmp[dkm2][dj-1][di] += c[di];
          }

          // +/- z
          if (zstages != 1 || dk != 0) {
            if (0 <= dj && dj < my) {
              tmp[1-dkm2][dj][di] += c[di];
            }
          }

        }

        if(0 < dk && 0 <= dj && dj < my && 0 < j && j < ny-1) {
          // write out completed values
          aligned_store_chunk(&out[id(i0, j, k-1)], tmp[1-dkm2][dj]);
        }

        if (zstages != 1 || dk != 0) {
          if(0 <= dj && dj < my) {
            // reset recently-written values to zero
            tmp[1-dkm2][dj] = {};

            // so we can add the +z contribution to the new cell
            #pragma unroll
            for (int di = 0; di < mx; di++) {
              tmp[1-dkm2][dj][di] += c[di];
            }
          }
        }

      }

    }

  }

}

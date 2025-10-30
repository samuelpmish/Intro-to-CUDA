#include "vector.hpp"
#include "timer.hpp"
#include "span.hpp"

#include <iostream>
#include <cusparse.h>

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

cusparseHandle_t handle = NULL;
cusparseSpMatDescr_t matA;
cusparseDnVecDescr_t vec_x, vec_y;
double alpha = 1.0;
double beta = 0.0;
void * d_buffer = nullptr;
size_t buffer_size = 0;

int * d_offsets;
int * d_columns;
double * d_values;
double * d_x;
double * d_y;

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

void initialize_CSR_matrix(int n) {

  std::vector< int > h_offsets(n * n * n + 1);

  std::vector< int > h_columns;
  h_columns.reserve(n * n * n * 7);

  std::vector< double > h_values;
  h_values.reserve(n * n * n * 7);

  auto id = [&](int x, int y, int z) {
    return z * n * n + y * n + x;
  };

  auto on_bdr = [&](int i) { 
    return (i == 0) || (i == (n-1)); 
  };

  auto push = [&](int col, double val) { 
    h_values.push_back(val);
    h_columns.push_back(col);
  };

  // jacobi iteration matrix
  h_offsets[0] = 0;
  for (int k = 0; k < n; k++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        int cell_id = id(i, j, k);
        if (on_bdr(i) || on_bdr(j) || on_bdr(k)) {
          push(cell_id, 1.0);
        } else {
          push(id(i+1, j  , k  ), +1.0);
          push(id(i-1, j  , k  ), +1.0);
          push(id(i  , j+1, k  ), +1.0);
          push(id(i  , j-1, k  ), +1.0);
          push(id(i  , j  , k+1), +1.0);
          push(id(i  , j  , k-1), +1.0);
        }
        h_offsets[cell_id+1] = h_values.size();
      } 
    } 
  } 

  int ndof = n * n * n;
  int nnz = h_offsets.back();

  std::cout << "ndof: " << ndof << ", nnz: " << nnz << std::endl;

  cudaMalloc(&d_offsets, (ndof + 1) * sizeof(int));
  cudaMalloc(&d_columns, nnz * sizeof(int));
  cudaMalloc(&d_values, nnz * sizeof(double));
  cudaMalloc(&d_x, ndof * sizeof(double));
  cudaMalloc(&d_y, ndof * sizeof(double));

  cudaMemcpy(d_offsets, &h_offsets[0], (ndof + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_columns, &h_columns[0], nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, &h_values[0], nnz * sizeof(double), cudaMemcpyHostToDevice);

  CHECK_CUSPARSE(cusparseCreate(&handle));
  CHECK_CUSPARSE(cusparseCreateCsr(&matA, ndof, ndof, nnz,
                                   d_offsets, d_columns, d_values,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  CHECK_CUSPARSE(cusparseCreateDnVec(&vec_x, ndof, d_x, CUDA_R_64F));
  CHECK_CUSPARSE(cusparseCreateDnVec(&vec_y, ndof, d_y, CUDA_R_64F));

  // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vec_x, &beta, vec_y, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size))
  cudaMalloc(&d_buffer, buffer_size);

}

void cleanup() {

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE(cusparseDestroySpMat(matA));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vec_x));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vec_y));
  CHECK_CUSPARSE(cusparseDestroy(handle));

  cudaFree(d_offsets);
  cudaFree(d_columns);
  cudaFree(d_values);
  cudaFree(d_x);
  cudaFree(d_y);

}

int main() {

  int n = 256;
  int max_iterations = 200;
  double tolerance = 1.0e-2;

  initialize_CSR_matrix(n);

  // solution = 0 on the boundary, and we 
  // have a Dirac delta source on the interior
  std::vector<double> rhs(n * n * n, 0.0);
  rhs[(n / 2) * n * n + (n / 2) * n + (n / 2)] = 1.0;
  gpu::vector b = rhs;

  auto M = [&](const gpu::vector & x){
    gpu::vector Mx(x.size()); 

    // update pointers and execute SpMV
    CHECK_CUSPARSE(cusparseDnVecSetValues(vec_x, x.ptr));
    CHECK_CUSPARSE(cusparseDnVecSetValues(vec_y, Mx.ptr));
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vec_x, &beta, vec_y, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, d_buffer)); 

    return Mx;
  };

  timer stopwatch;

  // iterative solve by Jacobi method
  stopwatch.start();
  gpu::vector x(std::vector<double>(n * n * n, 0.0));
  for (int k = 0; k < max_iterations; k++) {
    x = (M(x) - b) / 6.0;
    double norm_r = residual_norm(x, b, n);
    std::cout << k << " " << norm_r << std::endl;
    if (norm_r < tolerance) {
      break;
    }
  }
  cudaDeviceSynchronize();
  stopwatch.stop();

  std::cout << "finished in " << stopwatch.elapsed() * 1000.0f << " ms" << std::endl;

  cleanup();

}
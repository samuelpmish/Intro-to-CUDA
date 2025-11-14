#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/zip_iterator.h>

#include "random.hpp"

struct tuple_less {
    __host__ __device__
    bool operator()(const thrust::tuple<int, int>& a, const thrust::tuple<int, int>& b) const {
        int arow = thrust::get<0>(a), acol = thrust::get<1>(a);
        int brow = thrust::get<0>(b), bcol = thrust::get<1>(b);
        return (arow < brow) || (arow == brow && acol < bcol);
    }
};

struct tuple_equal {
    __host__ __device__
    bool operator()(const thrust::tuple<int, int>& a, const thrust::tuple<int, int>& b) const {
        return thrust::get<0>(a) == thrust::get<0>(b) && thrust::get<1>(a) == thrust::get<1>(b);
    }
};

struct Resistor {
    int node[2];
    double resistance;
};

__global__ void emit_coo_values(int * row, int * col, double * values, const Resistor * resistors, int num_resistors) {

    // TODO: emit 4 COO-triples (row, col, value) for each resistor

}

int main() {

    int num_nodes = 1 << 15;
    int num_resistors = 1 << 21;

    std::vector< Resistor > h_resistors(num_resistors);
    for (int i = 0; i < num_resistors; i++) {
        h_resistors[i] = Resistor{
            {random_integer(0, num_nodes-1), random_integer(0, num_nodes-1)},
            random_real(1.0, 10.0)
        };
    }

    int * d_coo_rows;
    int * d_coo_cols;
    double * d_coo_values;
    Resistor * d_resistors;

    cudaMalloc(&d_coo_rows, 4 * num_resistors * sizeof(int));
    cudaMalloc(&d_coo_cols, 4 * num_resistors * sizeof(int));
    cudaMalloc(&d_coo_values, 4 * num_resistors * sizeof(double));
    cudaMalloc(&d_resistors, num_resistors * sizeof(Resistor));

    cudaMemcpy(d_resistors, &h_resistors[0], sizeof(Resistor) * num_resistors, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (num_resistors + block - 1) / block;
    emit_coo_values<<< grid, block >>>(d_coo_rows, d_coo_cols, d_coo_values, d_resistors, num_resistors);

    int ncoo = 4 * num_resistors;

    // sort COO entries by (row, col) to make repeated entries adjacent
    auto key_zip = thrust::make_zip_iterator(thrust::make_tuple(
        d_coo_rows, 
        d_coo_cols
    ));
    thrust::sort_by_key(thrust::device, key_zip, key_zip + ncoo, d_coo_values, tuple_less());

    // reduce by (row,col) key to combine values 
    int * d_coo_rows_dedup;
    int * d_csr_col_ind;
    double * d_csr_values;

    cudaMalloc(&d_coo_rows_dedup, 4 * num_resistors * sizeof(int));
    cudaMalloc(&d_csr_col_ind, 4 * num_resistors * sizeof(int));
    cudaMalloc(&d_csr_values, 4 * num_resistors * sizeof(double));

    auto key_dedup_zip = thrust::make_zip_iterator(thrust::make_tuple(d_coo_rows_dedup, d_csr_col_ind));

    auto reduce_end = thrust::reduce_by_key(
        thrust::device,
        key_zip, key_zip + ncoo, d_coo_values,  // input keys and values
        key_dedup_zip, d_csr_values,            // output keys and values
        tuple_equal(), thrust::plus<double>()); // predicate, reduction

    int nnz = reduce_end.first - key_dedup_zip;

    std::cout << "num COO entries: " << ncoo;
    std::cout << ", num nonzero CSR entries: " << nnz << std::endl;

    int * d_csr_row_ptr;
    cudaMalloc(&d_csr_row_ptr, (num_nodes + 1) * sizeof(int));
    thrust::sequence(thrust::device, d_csr_row_ptr, d_csr_row_ptr + num_nodes + 1, 0);

    // binary search to find where d_coo_rows_dedup values change
    thrust::lower_bound(
        thrust::device, 
        d_coo_rows_dedup, d_coo_rows_dedup + nnz,
        d_csr_row_ptr, d_csr_row_ptr + num_nodes,
        d_csr_row_ptr);

    //---------------------------------------//
    // do something with the matrix entries! //
    //---------------------------------------//

    cudaFree(d_coo_rows_dedup);
    cudaFree(d_csr_row_ptr);
    cudaFree(d_csr_col_ind);
    cudaFree(d_csr_values);

    cudaFree(d_coo_rows);
    cudaFree(d_coo_cols);
    cudaFree(d_coo_values);

}
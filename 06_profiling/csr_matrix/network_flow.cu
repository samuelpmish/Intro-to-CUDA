#include <vector>
#include <iostream>

#include "timer.hpp"
#include "random.hpp"

struct Arc {
    uint64_t node_1;
    uint64_t node_2;
    double cost;
    double capacity;
};

__global__ void emit_coo_values(int * row, int * col, double * values, const Arc * arcs, int num_arcs) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_arcs) {
        auto [i, j, cost, cap] = arcs[tid];

        row[2 * tid + 0] = i;
        col[2 * tid + 0] = tid;
        values[2 * tid + 0] = -1.0;

        row[2 * tid + 1] = j;
        col[2 * tid + 1] = tid;
        values[2 * tid + 1] = +1.0;
    }

}

int main() {

    int num_nodes = 1 << 15;
    int num_arcs = 1 << 21;

    std::vector< Arc > h_arcs(num_arcs);
    for (int i = 0; i < num_arcs; i++) {
        h_arcs[i] = Arc{
            uint64_t(random_integer(0, num_nodes-1)), 
            uint64_t(random_integer(0, num_nodes-1)),
            random_real(1.0, 10.0),
            random_real(1.0, 10.0)
        };
    }

    Arc * d_arcs;
    int * d_coo_rows;
    int * d_coo_cols;
    double * d_coo_values;

    cudaMalloc(&d_arcs, num_arcs * sizeof(Arc));
    cudaMalloc(&d_coo_rows, 2 * num_arcs * sizeof(int));
    cudaMalloc(&d_coo_cols, 2 * num_arcs * sizeof(int));
    cudaMalloc(&d_coo_values, 2 * num_arcs * sizeof(double));

    cudaMemcpy(d_arcs, &h_arcs[0], sizeof(Arc) * num_arcs, cudaMemcpyHostToDevice);

    float time_ms = kernel_time_in_ms([&](){
        int block = 256;
        int grid = (num_arcs + block - 1) / block;
        emit_coo_values<<< grid, block >>>(d_coo_rows, d_coo_cols, d_coo_values, d_arcs, num_arcs);
    });

    std::cout << "emit_coo_values time: " << time_ms << " ms" << std::endl;

    cudaFree(d_arcs);
    cudaFree(d_coo_rows);
    cudaFree(d_coo_cols);
    cudaFree(d_coo_values);

}
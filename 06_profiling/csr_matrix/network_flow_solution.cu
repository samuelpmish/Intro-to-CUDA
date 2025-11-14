#include <vector>
#include <iostream>

#include "timer.hpp"
#include "random.hpp"

// just the connectivity info: 
// store the cost / capacity values in separate arrays
struct Edge {
    int node_1;
    int node_2;
};

__global__ void emit_coo_values(int * row, int * col, double * values, const Edge * edges, int num_arcs) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_arcs) {
        auto [i, j] = edges[tid];

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

    std::vector< Edge > h_edges(num_arcs);
    for (int i = 0; i < num_arcs; i++) {
        h_edges[i] = Edge{
            random_integer(0, num_nodes-1), 
            random_integer(0, num_nodes-1)
        };
    }

    Edge * d_edges;
    int * d_coo_rows;
    int * d_coo_cols;
    double * d_coo_values;

    cudaMalloc(&d_edges, num_arcs * sizeof(Edge));
    cudaMalloc(&d_coo_rows, 2 * num_arcs * sizeof(int));
    cudaMalloc(&d_coo_cols, 2 * num_arcs * sizeof(int));
    cudaMalloc(&d_coo_values, 2 * num_arcs * sizeof(double));

    cudaMemcpy(d_edges, &h_edges[0], sizeof(Edge) * num_arcs, cudaMemcpyHostToDevice);

    float time_ms = kernel_time_in_ms([&](){
        int block = 256;
        int grid = (num_arcs + block - 1) / block;
        emit_coo_values<<< grid, block >>>(d_coo_rows, d_coo_cols, d_coo_values, d_edges, num_arcs);
    });

    std::cout << "emit_coo_values time: " << time_ms << " ms" << std::endl;

    cudaFree(d_edges);
    cudaFree(d_coo_rows);
    cudaFree(d_coo_cols);
    cudaFree(d_coo_values);

}
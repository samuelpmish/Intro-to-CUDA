#include <cstdio>
#include <cinttypes>

static constexpr int n = 10;

void cpu_function(double * data) {
    for (int i = 0; i < n; i++) {
        printf("%d: %f\n", i, data[i]);
    }
}

__global__ void gpu_kernel(double * data) {
    for (int i = 0; i < n; i++) {
        printf("%d: %f\n", i, data[i]);
    }
}

int main() {

    double * h_data_malloc;
    double * h_data_pinned;
    double * d_data_managed;
    double * d_data;

    // allocate memory
    uint32_t num_bytes = sizeof(double) * n;
    h_data_malloc = (double *)malloc(num_bytes);
    cudaMallocHost(&h_data_pinned, num_bytes);
    cudaMalloc(&d_data, num_bytes);
    cudaMallocManaged(&d_data_managed, num_bytes);

    // initialize host memory 
    for (int i = 0; i < n; i++) {
        h_data_malloc[i] = i * i + 1;
    }

    // copy host memory to other buffers
    cudaMemcpy(h_data_pinned, h_data_malloc, num_bytes, cudaMemcpyHostToHost);
    cudaMemcpy(d_data, h_data_malloc, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_managed, h_data_malloc, num_bytes, cudaMemcpyHostToDevice);

/******************************************************************************/

    // try accessing each of the memory buffers from the CPU and GPU -- what works and what doesn't?
    
    // e.g. accessing malloc buffer from cpu_function
    cpu_function(h_data_malloc);

/******************************************************************************/

    // deallocate memory
    free(h_data_malloc);
    cudaFreeHost(h_data_pinned);
    cudaFree(d_data_managed);
    cudaFree(d_data);

}
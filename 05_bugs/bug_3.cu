// try running with:
// compute-sanitizer --tool initcheck --track-unused-memory ./bug_3

#include <cstdio>
#include <vector>
 
__global__ void increment(float * data) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  data[tid]++;
}
 
int main(int argc, char **argv) {
  int n = 64;

  float * d_data;
  cudaMalloc(&d_data, sizeof(double) * n);
  cudaMemset(d_data, 0, sizeof(float) * n);
 
  int threads_per_block = 32;
  int blocks = n / threads_per_block;
  increment<<< blocks, threads_per_block >>>(d_data);

  std::vector< float > h_data(n);
  cudaMemcpy(&h_data[0], d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

  printf("After : Vector 0, 1 .. n-1: %f %f .. %f\n", h_data[0], h_data[1], h_data[n-1]);
 
  cudaFree(d_data);
}
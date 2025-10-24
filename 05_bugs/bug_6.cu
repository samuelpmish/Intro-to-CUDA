// try running with:
// compute-sanitizer --tool synccheck ./bug_6

#include <cstdio>
#include <vector>

static constexpr int n = 512;
 
__global__ void reduce(float * data, float * sum) {
  __shared__ float shmem[n];

  shmem[threadIdx.x] = data[threadIdx.x] + threadIdx.x; 

  for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shmem[threadIdx.x] += shmem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *sum = shmem[0];
  }
}
 
int main(int argc, char **argv) {

  float * d_data;
  cudaMalloc(&d_data, sizeof(float) * n);
  cudaMemset(d_data, 0, sizeof(float) * n);

  float * d_sum;
  cudaMalloc(&d_sum, sizeof(float));
  cudaMemset(d_sum, 0, sizeof(float));
 
  reduce<<< 1, n >>>(d_data, d_sum);

  float h_sum;
  cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

  printf("h_sum: %f, expected: %d\n", h_sum, (n * (n - 1)) / 2);
 
  cudaFree(d_data);
  cudaFree(d_sum);
}
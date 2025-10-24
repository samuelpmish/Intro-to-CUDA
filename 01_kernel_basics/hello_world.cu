#include <cstdio>

__global__ void kernel() {
    printf("hello world, from the GPU\n");
}

int main() {
    printf("hello world, from the CPU\n");

    // launch our CUDA kernel with 1 block, and 1 thread / block
    kernel<<<1,1>>>();
}
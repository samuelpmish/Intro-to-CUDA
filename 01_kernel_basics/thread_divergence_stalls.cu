#include <cstdio>
#include <iostream>
#include <cinttypes>

__global__ void kernel(float seed) {

    float total = seed + threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t mask;
    
    mask = __activemask();
    if (threadIdx.x % 32 == 0 && mask != 0xFFFFFFFF) {
        printf("before switch %d, %d, %#010x\n", blockIdx.x, threadIdx.x, mask);
    }

    switch(threadIdx.x / 8) {
        case 0:
            total = sin(total);
            break;
        case 1:
            for (int i = 0; i < 1000; i++) {
                total = cos(total);
            }
            break;
        case 2:
            total += 3.0;
            break;
        case 3:
            total *= 4.2;
            break;
    }

    mask = __activemask();
    if (threadIdx.x % 32 == 0 && mask != 0xFFFFFFFF) {
        printf("after switch %d, %d, %#010x\n", blockIdx.x, threadIdx.x, mask);
        printf("%f", total);
    }

}

int main() {
    kernel<<<100000,128>>>(3.14);
    cudaDeviceSynchronize();
}
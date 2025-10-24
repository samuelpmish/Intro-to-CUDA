#include <cstdio>

__global__ void kernel() {
    switch(threadIdx.x % 4) {

        //✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌
        case 0:
            for (int i = 0; i < 2; i++) {
                printf("case 0: hello from thread %d, i = %d\n", threadIdx.x, i);
                __nanosleep(1000);
            }
            break;

        //❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌
        case 1:
            for (int i = 0; i < 3; i++) {
                printf("case 1: hello from thread %d, i = %d\n", threadIdx.x, i);
                __nanosleep(1000);
            }
            break;

        //❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌
        case 2:
            for (int i = 0; i < 4; i++) {
                printf("case 2: hello from thread %d, i = %d\n", threadIdx.x, i);
                __nanosleep(1000);
            }
            break;

        //❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅❌❌❌✅
        case 3:
            for (int i = 0; i < 5; i++) {
                printf("case 3: hello from thread %d, i = %d\n", threadIdx.x, i);
                __nanosleep(1000);
            }
            break;
    }
}

int main() {
    // launch our CUDA kernel with 1 block, and 8 threads / block
    kernel<<<1,8>>>();
    cudaDeviceSynchronize();
}
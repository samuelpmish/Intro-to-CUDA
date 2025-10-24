#include <cstdio>

#define NUM_ELEMENTS 23

__global__ void correct(const float expected) {

    float val = threadIdx.x;

    unsigned mask = __ballot_sync(0xFFFFFFFF, threadIdx.x < NUM_ELEMENTS);
    if (threadIdx.x < NUM_ELEMENTS) { 
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }
    }

    if (threadIdx.x == 0 && expected != val) {
        printf("%f\n", val);
    }

}

__global__ void incorrect(const float expected) {

    float val = threadIdx.x;

    if (threadIdx.x < NUM_ELEMENTS) { 
        unsigned mask = __activemask(); 
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }
    }

    if (threadIdx.x == 0 && expected != val) {
        printf("%f\n", val);
    }

}

__global__ void incorrect_2(const float expected) {

    __shared__ float shmem[32];

    shmem[threadIdx.x] = threadIdx.x;

    for (int offset = 16; offset > 0; offset /= 2) {
        if (threadIdx.x + offset < NUM_ELEMENTS) { 
            shmem[threadIdx.x] += shmem[threadIdx.x + offset];
        }
    }

    if (threadIdx.x == 0 && expected != shmem[0]) {
        printf("%f\n", shmem[0]);
    }

}

int main() {
    float expected = 0.0f;
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        expected += i;
    }

    correct<<<1,32>>>(expected);
    incorrect<<<1,32>>>(expected);
    incorrect_2<<<1,32>>>(expected);
    cudaDeviceSynchronize();
}
#include <cstdio>
#include <cstdlib>

__global__ void long_kernel() {
    for (int i = 0; i < 1e2; ++i) {
        __nanosleep(1e6);
    }
}

float run_test(int num_blocks) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    long_kernel<<<num_blocks, 1>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float duration;
    cudaEventElapsedTime(&duration, start, stop);

    return duration;
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        printf("must specify number of blocks to run\n");
        exit(0);
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_SMs = prop.multiProcessorCount;
    int num_blocks = atoi(argv[1]);

    printf("number of blocks: %d, blocks / SM: %f, time = %f\n", 
        num_blocks, 
        float(num_blocks) / num_SMs,
        run_test(num_blocks)
    );

}
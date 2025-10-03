#pragma once

#include <cstdio>
#include <cuda.h>

namespace impl {
    inline void cuda_check(cudaError_t code, const char *file, int line) {
        if (code != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
            exit(code);
        }
    }
}

#define CUDA_CHECK(call) { impl::cuda_check((call), __FILE__, __LINE__); }

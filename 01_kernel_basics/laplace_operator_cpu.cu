#include <vector>
#include <iostream>

#include "timer.hpp"

struct span3D{

    static constexpr int dim = 3;

    __host__ __device__ span3D(float * data, const int (&shape)[dim]) {
        data_ = data;

        shape_[0] = shape[0];
        shape_[1] = shape[1];
        shape_[2] = shape[2];

        stride_[1] = shape_[2];
        stride_[0] = shape_[1] * shape_[2];
    }

    __host__ __device__ float & operator()(int i, int j, int k) {
        return data_[i * stride_[0] + j * stride_[1] + k];
    }

    __host__ __device__ const float & operator()(int i, int j, int k) const {
        return data_[i * stride_[0] + j * stride_[1] + k];
    }

    float * data_;
    int shape_[3];
    int stride_[2];

};

void laplace_operator(span3D output, const span3D input) {
    for (int k = 1; k < input.shape_[2] - 1; ++k) {
        for (int j = 1; j < input.shape_[1] - 1; ++j) {
            for (int i = 1; i < input.shape_[0] - 1; ++i) {
                output(i, j, k) = input(i+1, i  , i  ) +
                                  input(i-1, i  , i  ) +
                                  input(i  , i+1, i  ) +
                                  input(i  , i-1, i  ) +
                                  input(i  , i  , i+1) +
                                  input(i  , i  , i-1) - 
                                  input(i  , i  , i  ) * 6.0f;
            }
        }
    }
}

int main() {

    int n = 512;

    std::vector<float> h_in(n * n * n);
    std::vector<float> h_out(n * n * n);

    span3D input(&h_in[0], {n, n, n});
    span3D output(&h_out[0], {n, n, n});

    timer stopwatch;

    stopwatch.start();
    laplace_operator(output, input);
    stopwatch.stop();

    float time = stopwatch.elapsed();

    std::cout << "time: " << time * 1000.0f << " ms " << std::endl;

}
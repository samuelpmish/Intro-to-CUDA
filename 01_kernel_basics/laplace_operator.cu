#include <iostream>

#include "timer.hpp"

struct span3D{

    static constexpr int dim = 3;

    span3D(float * data, const int (&shape)[dim]) {
        data_ = data;

        shape_[0] = shape[0];
        shape_[1] = shape[1];
        shape_[2] = shape[2];

        stride_[1] = shape_[2];
        stride_[0] = shape_[1] * shape_[2];
    }

    float & operator()(int i, int j, int k) {
        return data_[i * stride_[0] + j * stride_[1] + k];
    }

    const float & operator()(int i, int j, int k) const {
        return data_[i * stride_[0] + j * stride_[1] + k];
    }

    float * data_;
    int shape_[3];
    int stride_[2];

};

__global__ void laplace_operator(span3D output, const span3D input) {

    // TODO: implement laplace operator calculation! 
    // see: laplace_operator_cpu for reference CPU implementation
    //      might also need to make minor changes to the span3D container

}


int main() {

    int n = 512;

    float * d_in;
    float * d_out;

    cudaMalloc(&d_in, sizeof(float) * n * n * n);
    cudaMalloc(&d_out, sizeof(float) * n * n * n);

    span3D input(d_in, {n, n, n});
    span3D output(d_out, {n, n, n});

    timer stopwatch;

    dim3 block = {8, 8, 8};
    dim3 grid = {n / block.x, n / block.y, n / block.z};

    stopwatch.start();
    laplace_operator<<< grid, block >>>(output, input);
    cudaDeviceSynchronize();
    stopwatch.stop();

    float time = stopwatch.elapsed();

    std::cout << "time: " << time * 1000.0f << " ms " << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

}
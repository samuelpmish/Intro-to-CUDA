#include <vector>
#include <iostream>

#include "span.hpp"
#include "timer.hpp"

void laplace_operator(span3D<float> output, const span3D<float> input) {
  for (int k = 0; k < input.shape_[2]; ++k) {
    for (int j = 0; j < input.shape_[1]; ++j) {
      for (int i = 0; i < input.shape_[0]; ++i) {
        bool in_bounds = (0 < i) && (i < input.shape_[2] - 1) && 
                         (0 < j) && (j < input.shape_[1] - 1) && 
                         (0 < k) && (k < input.shape_[0] - 1);
        if (in_bounds) {
          output(k, j, i) = input(k + 1, j    , i    ) 
                          + input(k - 1, j    , i    )
                          + input(k    , j + 1, i    )
                          + input(k    , j - 1, i    )
                          + input(k    , j    , i + 1) 
                          + input(k    , j    , i - 1) 
                          - input(k    , j    , i) * 6.0f;
        } else {
          output(k, j, i) = input(k, j, i);
        }
      }
    }
  }
}

int main() {

    int n = 512;

    std::vector<float> h_in(n * n * n);
    std::vector<float> h_out(n * n * n);

    span3D<float> input(&h_in[0], {n, n, n});
    span3D<float> output(&h_out[0], {n, n, n});

    timer stopwatch;

    stopwatch.start();
    laplace_operator(output, input);
    stopwatch.stop();

    float time = stopwatch.elapsed();

    std::cout << "time: " << time * 1000.0f << " ms " << std::endl;

}
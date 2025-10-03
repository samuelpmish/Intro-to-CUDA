#include <vector>
#include <iostream>

#include "timer.hpp"

void saxpy(const float a, const float * x, float * y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {

    // we'll stick to powers of two so everything divides evenly
    int n = 1 << 22;

    float a = 1.0;
    std::vector<float> x(n, 1.0);
    std::vector<float> y(n, 2.0);

    timer stopwatch;

    stopwatch.start();
    saxpy(a, &x[0], &y[0], n);
    stopwatch.stop();

    float time = stopwatch.elapsed();
    uint32_t num_bytes = n * sizeof(float) * 3; // 2 reads + 1 write

    std::cout << "time: " << time * 1000.0f << " ms " << std::endl;
    std::cout << "effective memory bandwidth: " << (num_bytes / time) * 1.0e-9f << " GB/s " << std::endl;

}
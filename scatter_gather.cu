#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <random>
#include <algorithm>

#include <chrono>

class timer {
  typedef std::chrono::high_resolution_clock::time_point time_point;
  typedef std::chrono::duration<double>                  duration_type;

public:
  void start() { then = std::chrono::high_resolution_clock::now(); }
  void stop() { now = std::chrono::high_resolution_clock::now(); }
  double elapsed() { return std::chrono::duration_cast<duration_type>(now - then).count(); }

private:
  time_point then, now;
};

template < typename callable >
double time(callable f, int n = 1) {
  timer stopwatch;
  stopwatch.start();
  for (int i = 0; i < n; i++) {
    f();
  }
  stopwatch.stop();
  return stopwatch.elapsed();
}

void comparison_test(int n, double radius) {

    int num_iterations = 10;

    std::vector< std::pair< int, double > > pairs(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-radius, radius);

    for (int i = 0; i < n; i++) {
        pairs[i] = {i, i + dis(gen)};
    }

    std::sort(pairs.begin(), pairs.end(), [](auto & a, auto & b) { return a.second < b.second;});

    std::vector< int > iota(n);
    std::vector< int > permutation(n);
    std::vector< int > inv_permutation(n);
    for (int i = 0; i < n; i++) {
        iota[i] = i;
        permutation[i] = pairs[i].first;
        inv_permutation[pairs[i].first] = i;
    }


    thrust::device_vector< float > input(iota);
    thrust::device_vector< float > output1(n);
    thrust::device_vector< float > output2(n);

    thrust::device_vector<int> d_permutation(permutation.begin(), permutation.end());
    thrust::device_vector<int> d_inv_permutation(inv_permutation.begin(), inv_permutation.end());

    float time1 = time([&](){
        thrust::scatter(thrust::device,
                        input.begin(), input.end(),
                        d_permutation.begin(), output1.begin());
    }, num_iterations) / num_iterations;

    float time2 = time([&](){
        thrust::gather(thrust::device,
                       d_inv_permutation.begin(), d_inv_permutation.end(),
                       input.begin(), output2.begin());
    }, num_iterations) / num_iterations;

    //for (int i = 0; i < 5; i++) {
    //    std::cout << output1[i] << " " << output2[i] << std::endl;
    //}

    std::cout << radius << ", " << time1 << ", " << time2 << std::endl;

}

int main() {

    int n = 10'000'000;
    comparison_test(n, 0.0);
    for (uint32_t i = 1; i < 100 * n; i <<= 3) {
        double radius = i;
        comparison_test(n, radius);
    }

}


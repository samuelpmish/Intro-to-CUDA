#pragma once

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

#if __CUDACC__
template < typename callable >
float kernel_time_in_ms(callable f) {

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  f();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float time_ms;
  cudaEventElapsedTime(&time_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time_ms;

}
#endif

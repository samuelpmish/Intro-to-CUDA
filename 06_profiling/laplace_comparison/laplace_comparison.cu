#include <vector>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include "parse.hpp"
#include "chunk.hpp"
#include "error_checking.hpp"

#include "laplace_original.hpp"
#include "laplace_vectorized.hpp"
#include "laplace_tiled.hpp"

template < int m >
struct constexpr_int { constexpr operator int() { return m; } };

dim3 make_grid(uint32_t n, uint32_t bx, uint32_t by, uint32_t bz) {
  return dim3{
    (n + bx - 1) / bx,
    (n + by - 1) / by,
    (n + bz - 1) / bz
  };
}

template < typename T >
void run_tests(int n, int num_iterations, std::array<uint32_t,3> blocksz, bool print_header) {

  cudaEvent_t start;
  cudaEvent_t end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  size_t num_elements = n * n * n;

  T * d_in;
  T * d_out;
  int * d_errors;
  CUDA_CHECK(cudaMalloc(&d_in, num_elements * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_out, num_elements * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_errors, sizeof(int)));

  std::vector< T > h_in(num_elements);
  std::vector< T > h_out(num_elements);

  double a =  0.05;
  double b =  0.03;
  double c = -0.02;
  double answer = 2 * (a + b + c);

  for (int k = 0; k < n; k++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        int tid = i + j*n + k*n*n;
	      T x = T(i) / n;
	      T y = T(j) / n;
	      T z = T(k) / n;
        h_in[tid] = a * x * x + b * y * y + c * z * z;
      }
    }
  }

  CUDA_CHECK(cudaMemcpy(d_in, &h_in[0], sizeof(T) * num_elements, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, num_elements * sizeof(T)));

  dim3 block = {blocksz[0], blocksz[1], blocksz[2]};

  T epsilon;
  if (sizeof(T) == 4) {
    epsilon = 1.0e-5;
  } else {
    epsilon = 1.0e-9;
  }

  auto time_kernel_ms = [&](auto launch, bool check = true) {

    // don't time the first one
    cudaMemset(d_out, 0, sizeof(T) * num_elements);
    launch();

    // --- Launch the kernel repeatedly for profiling ---
    float total_time_ms = 0.0f;
    for (int it = 0; it < num_iterations; ++it) {

      cudaEventRecord(start);
      launch();
      cudaEventRecord(end);
      CUDA_CHECK(cudaDeviceSynchronize());

      float time_ms;
      cudaEventElapsedTime(&time_ms, start, end);
      total_time_ms += time_ms;
    }

    return total_time_ms / num_iterations;

  };

  using entry = std::tuple< std::string, float >;
  std::vector< entry > entries;
  entries.push_back({"   n", n});
  entries.push_back({"  bx", blocksz[0]});
  entries.push_back({"  by", blocksz[1]});
  entries.push_back({"  bz", blocksz[2]});

////////////////////////////////////////////////////////////////////////////////

  entries.push_back({"cudaMemcpy", time_kernel_ms([&](){
    cudaMemcpy(d_out, d_in, num_elements * sizeof(T), cudaMemcpyDeviceToDevice);
  }, false)});

////////////////////////////////////////////////////////////////////////////////

  entries.push_back({"original", time_kernel_ms([&](){
    dim3 grid = make_grid(n, block.x, block.y, block.z);
    laplace_original<<<grid, block>>>(d_out, d_in, n, n, n);
  })});

////////////////////////////////////////////////////////////////////////////////

  auto vectorized_entry = [&](auto mx){
    std::string label = std::string("vectorized") + std::to_string(mx); 
    float time_ms = time_kernel_ms([&](){
      dim3 grid = make_grid(n, block.x * mx, block.y, block.z);
      laplace_vectorized<mx><<<grid, block>>>(d_out, d_in, n, n, n);
    });

    return entry{label, time_ms};
  };

  entries.push_back(vectorized_entry(constexpr_int<2>{}));
  entries.push_back(vectorized_entry(constexpr_int<4>{}));

////////////////////////////////////////////////////////////////////////////////

  auto tiled_entry = [&](auto mx, auto my, auto mz){
    std::string label = std::string("tiled") + std::to_string(mx) + "x" + std::to_string(my) + "x" + std::to_string(mz); 
    float time_ms = time_kernel_ms([&](){
      dim3 grid = make_grid(n, block.x * mx, block.y * my, block.z * mz);
      laplace_tiled_3D<mx, my, mz><<<grid, block>>>(d_out, d_in, n, n, n);
    });

    return entry{label, time_ms};
  };

  entries.push_back(tiled_entry(constexpr_int<2>{}, constexpr_int<1>{}, constexpr_int<1>{}));
  entries.push_back(tiled_entry(constexpr_int<4>{}, constexpr_int<1>{}, constexpr_int<1>{}));
  entries.push_back(tiled_entry(constexpr_int<2>{}, constexpr_int<2>{}, constexpr_int<1>{}));
  entries.push_back(tiled_entry(constexpr_int<4>{}, constexpr_int<2>{}, constexpr_int<1>{}));
  entries.push_back(tiled_entry(constexpr_int<4>{}, constexpr_int<4>{}, constexpr_int<1>{}));
  entries.push_back(tiled_entry(constexpr_int<2>{}, constexpr_int<2>{}, constexpr_int<2>{}));
  entries.push_back(tiled_entry(constexpr_int<4>{}, constexpr_int<4>{}, constexpr_int<4>{}));

////////////////////////////////////////////////////////////////////////////////
  
  if (print_header) {
    for (int i = 0; i < entries.size(); i++) {
      std::cout << std::get<0>(entries[i]);
      if (i != entries.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << std::endl;
  }

  for (int i = 0; i < entries.size(); i++) {
    auto [label, value] = entries[i];
    std::cout << std::setw(label.size()) << value;
    if (i != entries.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << std::endl;

////////////////////////////////////////////////////////////////////////////////

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_errors);

  cudaEventDestroy(start);
  cudaEventDestroy(end);
  
}

int main(int argc, char* argv[]) {

  int n = 512;
  bool fp32 = false;
  bool no_header = false;
  int num_iterations = 1;
  std::array<uint32_t,3> block = {32, 8, 1};

  // parse CLI arguments
  parse(argc, argv, 
    std::tuple{std::string("-n"), &n}, 
    std::tuple{std::string("-iter"), &num_iterations}, 
    std::tuple{std::string("-fp32"), &fp32}, 
    std::tuple{std::string("-block"), &block},
    std::tuple{std::string("-noheader"), &no_header}
  );

  if (block[2] > 1) {
    std::cout << "note: several of the shared memory kernels assume block.z == 1" << std::endl;
  }

  if (n % 4 != 0) {
    std::cout << "note: several of the kernel implementations require padding x dimension to multiple of 4" << std::endl;
  }

  if (fp32) {
    run_tests< float >(n, num_iterations, block, !no_header);
  } else {
    run_tests< double >(n, num_iterations, block, !no_header);
  }

  return 0;

}

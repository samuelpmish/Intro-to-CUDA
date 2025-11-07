#pragma once

#include <cstdint> // for std::uintptr_t

template < typename T, int n >
struct alignas(sizeof(T) * n) chunk {
  T values[n];
  __host__ __device__ T & operator[](int i){ return values[i]; }
};

////////////////////////////////////////////////////////////////////////////////

#pragma region operators 

template < typename T, int n >
__host__ __device__ void operator+=(chunk<T,n> & x, const chunk<T,n> & y) {
  for (int i = 0; i < n; i++) {
    x.values[i] += y.values[i];
  }
}

template < typename T, int n >
__host__ __device__ chunk<T,n> operator+(const chunk<T,n> & x, const chunk<T,n> & y) {
  chunk<T,n> out;
  for (int i = 0; i < n; i++) {
    out.values[i] = x.values[i] + y.values[i];
  }
  return out;
}

template < typename T, int n >
__host__ __device__ chunk<T,n> operator-(const chunk<T,n> & x, const chunk<T,n> & y) {
  chunk<T,n> out;
  for (int i = 0; i < n; i++) {
    out.values[i] = x.values[i] - y.values[i];
  }
  return out;
}

template < typename T, int n >
__host__ __device__ chunk<T,n> operator*(const T scale, const chunk<T,n> & in) {
  chunk<T,n> out;
  for (int i = 0; i < n; i++) {
    out.values[i] = scale * in.values[i];
  }
  return out;
}

////////////////////////////////////////////////////////////////////////////////

// assumes ptr has proper alignment
template < int n, typename T >
__host__ __device__ chunk<T,n> aligned_load_chunk(const T * ptr) {
  chunk<T,n> c;
  c = *reinterpret_cast< const chunk<T,n> * >(ptr);
  return c;
}

// assumes ptr has proper alignment
template < int n, typename T >
__host__ __device__ void aligned_store_chunk(T * ptr, const chunk<T,n> & c) {
  *reinterpret_cast< chunk<T,n> * >(ptr) = c;
}
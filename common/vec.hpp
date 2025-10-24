#pragma once

#include <cmath>
#include <iostream>
#include <cinttypes>
#include <algorithm> // for std::{min,max}

template < uint32_t dim, typename T = double >
struct vec { 
  using data_type = T;
  static constexpr int dimension = dim;

  __host__ __device__ constexpr vec() : data{} {}

  __host__ __device__ constexpr vec(T a, T b) {
    static_assert(dim == 2, "vec(T, T) only supported for vec<2,T>");
    data[0] = a;
    data[1] = b;
  }

  __host__ __device__ constexpr vec(T a, T b, T c) {
    static_assert(dim == 3, "vec(T, T, T) only supported for vec<3,T>");
    data[0] = a;
    data[1] = b;
    data[2] = c;
  }

  __host__ __device__ constexpr vec(T a, T b, T c, T d) {
    static_assert(dim == 4, "vec(T, T, T, T) only supported for vec<4,T>");
    data[0] = a;
    data[1] = b;
    data[2] = c;
    data[3] = d;
  }

  __host__ __device__ constexpr vec(const T (&values)[dim]) {
    for (uint32_t i = 0; i < dim; i++) {
      data[i] = values[i];
    }
  }

  __host__ __device__ constexpr T & operator[](uint32_t i) { return data[i]; }
  __host__ __device__ constexpr const T & operator[](uint32_t i) const { return data[i]; }

  __host__ __device__ constexpr T & operator()(uint32_t i) { return data[i]; }
  __host__ __device__ constexpr const T & operator()(uint32_t i) const { return data[i]; }

  T data[dim];
};

template < typename T >
struct vec<1, T> { 
  using data_type = T;
  static constexpr int dimension = 1;

  __host__ __device__ constexpr vec() : data{} {}

  __host__ __device__ constexpr vec(T a) : data{a} {}

  // only allow implicit conversion to `T` when the vec is degenerate
  __host__ __device__ constexpr operator T() const { return data; }

  __host__ __device__ constexpr T & operator[](uint32_t) { return data; }
  __host__ __device__ constexpr const T & operator[](uint32_t) const { return data; }

  __host__ __device__ constexpr T & operator()(uint32_t) { return data; }
  __host__ __device__ constexpr const T & operator()(uint32_t) const { return data; }

  T data;
};

template < uint32_t dim, typename T = float >
vec(const T (&)[dim]) -> vec<dim, T>;

template < uint32_t dim >
using vecf = vec<dim, float>;

using vec1f = vec<1, float>;
using vec1 = vec<1, double>;

using vec2f = vec<2, float>;
using vec2 = vec<2, double>;

using vec3f = vec<3, float>;
using vec3 = vec<3, double>;

using vec4f = vec<4, float>;
using vec4 = vec<4, double>;

__host__ __device__ constexpr uint32_t dimension(double) { return 1; }

template < uint32_t dim, typename T >
__host__ __device__ constexpr uint32_t dimension(vec < dim, T >) { return dim; }

__host__ __device__ constexpr auto outer(const double & u, const double & v) {
  return u * v;
}

template <typename T, int n>
__host__ __device__ constexpr auto outer(const double & u, const vec<n,T> & v) {
  return u * v;
}

template <typename T, int n>
__host__ __device__ constexpr auto outer(const vec<n,T> & u, const double & v) {
  return u * v;
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim, typename S, typename T >
__host__ __device__ constexpr auto operator!=(const vec< dim, S > & u, const vec< dim, T > & v) {
  for (int i = 0; i < dim; i++) {
    if (u[i] != v[i]) return true;
  }
  return false;
}

template < uint32_t dim, typename S, typename T >
__host__ __device__ constexpr auto operator==(const vec< dim, S > & u, const vec< dim, T > & v) {
  for (uint32_t i = 0; i < dim; i++) {
    if (u[i] != v[i]) return false;
  }
  return true;
}

template < uint32_t n, typename T >
__host__ __device__ constexpr void operator*=(vec< n, T > & u, const T & v) {
  for (uint32_t i = 0; i < n; i++) u[i] *= v;
}

template < uint32_t n, typename T >
__host__ __device__ constexpr void operator+=(vec< n, T > & u, const vec< n, T > & v) {
  for (uint32_t i = 0; i < n; i++) u[i] += v[i];
}

template < uint32_t n, typename T >
__host__ __device__ constexpr void operator-=(vec< n, T > & u, const vec< n, T > & v) {
  for (uint32_t i = 0; i < n; i++) u[i] -= v[i];
}

template < uint32_t dim, typename S, typename T >
__host__ __device__ constexpr auto operator+(const vec< dim, S > & u, const vec< dim, T > & v) {
  vec< dim, decltype(S{} + T{}) > out{};
  for (uint32_t i = 0; i < dim; i++) {
    out[i] = u[i] + v[i];
  }
  return out; 
}

template < uint32_t dim, typename T >
__host__ __device__ constexpr auto operator-(const vec< dim, T > & v) {
  vec< dim, T > out{};
  for (uint32_t i = 0; i < dim; i++) {
    out[i] = -v[i];
  }
  return out; 
}

template < uint32_t dim, typename S, typename T >
__host__ __device__ constexpr auto operator-(const vec< dim, S > & u, const vec< dim, T > & v) {
  vec< dim, decltype(S{} - T{}) > out{};
  for (uint32_t i = 0; i < dim; i++) {
    out[i] = u[i] - v[i];
  }
  return out; 
}

template < uint32_t dim, typename S, typename T >
__host__ __device__ constexpr auto operator*(const vec< dim, S > & u, const vec< dim, T > & v) {
  vec< dim, decltype(S{} * T{}) > out{};
  for (uint32_t i = 0; i < dim; i++) {
    out[i] = u[i] * v[i];
  }
  return out; 
}

template < uint32_t dim, typename S, typename T >
__host__ __device__ constexpr auto operator/(const vec< dim, S > & u, const vec< dim, T > & v) {
  vec< dim, decltype(S{} / T{}) > out{};
  for (uint32_t i = 0; i < dim; i++) {
    out[i] = u[i] / v[i];
  }
  return out; 
}

////////////////////////////////////////////////////////////////////////////////

template < uint32_t dim, typename T >
__host__ __device__ constexpr auto operator*(const double & u, const vec< dim, T > & v) {
  vec< dim, T > out{};
  for (uint32_t i = 0; i < dim; i++) {
    out[i] = u * v[i];
  }
  return out; 
}

__host__ __device__ constexpr double dot(const double & u, const double & v) {
  return u * v;
}

template < uint32_t dim, typename T >
__host__ __device__ constexpr auto dot(const T & u, const vec< dim, T > & v) {
  vec< dim, T > out{};
  for (uint32_t i = 0; i < dim; i++) {
    out[i] = u * v[i];
  }
  return out; 
}

template < uint32_t dim, typename T >
__host__ __device__ constexpr auto operator*(const vec< dim, T > & v, const double & u) {
  vec< dim, T > out{};
  for (uint32_t i = 0; i < dim; i++) {
    out[i] = v[i] * u;
  }
  return out; 
}

template < uint32_t dim, typename T >
__host__ __device__ constexpr auto dot(const vec< dim, T > & v, const T & u) {
  vec< dim, T > out{};
  for (uint32_t i = 0; i < dim; i++) {
    out[i] = v[i] * u;
  }
  return out; 
}

template < uint32_t dim, typename T >
__host__ __device__ constexpr auto operator/(const double & u, const vec< dim, T > & v) {
  vec< dim, T > out{};
  for (uint32_t i = 0; i < dim; i++) {
    out[i] = u / v[i];
  }
  return out;
}

template < uint32_t dim, typename T >
__host__ __device__ constexpr auto operator/(const vec< dim, T > & v, const double & u) {
  vec< dim, T > out{};
  for (uint32_t i = 0; i < dim; i++) {
    out[i] = v[i] / u;
  }
  return out;
}

template < uint32_t dim, typename T >
__host__ __device__ constexpr auto norm_squared(const vec< dim, T > & v) {
  T out{};
  for (uint32_t i = 0; i < dim; i++) {
    out += v[i] * v[i];
  }
  return out; 
}

template < uint32_t dim, typename T >
__host__ __device__ constexpr auto norm(const vec< dim, T > & v) {
  T out{};
  for (uint32_t i = 0; i < dim; i++) {
    out += v[i] * v[i];
  }
  return std::sqrt(out); 
}

template < uint32_t dim, typename T >
__host__ __device__ constexpr auto normalize(const vec< dim, T > & v) {
  T norm_v = norm(v);
  if (norm_v != 0) {
    return v / norm_v;
  } else {
    return vec< dim, T >{};
  }
}

template < uint32_t dim, typename S, typename T >
__host__ __device__ constexpr auto dot(const vec< dim, S > & u, const vec< dim, T > & v) {
  decltype(S{} / T{}) total{};
  for (uint32_t i = 0; i < dim; i++) {
    total += u[i] * v[i];
  }
  return total;
}

// vector-vector products
template <typename T>
__host__ __device__ constexpr vec<2,T> cross(const vec<2,T> & u) {
  return vec<2,T>{-u[1], u[0]};
}

template <typename T>
__host__ __device__ constexpr vec<3,T> cross(const vec<3,T> & u, const vec<3,T> & v) {
  return vec<3,T>{
    u[1] * v[2] - u[2] * v[1], 
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0] 
  };
}

////////////////////////////////////////////////////////////////////////////////

template < typename T >
__host__ __device__ constexpr vec<2,T> xy(const vec<3,T> & v) { return {v[0], v[1]}; }

template < typename T >
__host__ __device__ constexpr vec<2,T> xy(const vec<2,T> & v) { return v; }

template < typename T >
__host__ __device__ constexpr vec<3,T> xyz(const vec<2,T> & v) { return {v[0], v[1], T{}}; }

template < typename T >
__host__ __device__ constexpr vec<3,T> xyz(const vec<3,T> & v) { return v; }

template < typename T >
__host__ __device__ constexpr vec<3,T> xyz(const vec<4,T> & v) { return {v[0], v[1], v[2]}; }

////////////////////////////////////////////////////////////////////////////////

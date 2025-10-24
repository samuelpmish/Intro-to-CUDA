#include <array>
#include <cstddef>
#include <iterator>
#include <algorithm>

template <typename T>
class pinned_allocator {
 public:
  using value_type = T;
  using pointer = value_type*;
  using size_type = std::size_t;

  pinned_allocator() noexcept = default;

  template <typename U>
  pinned_allocator(pinned_allocator<U> const&) noexcept {}

  auto allocate(size_type n, const void* = 0) -> value_type* {
      value_type * tmp;
      auto error = cudaMallocHost((void**)&tmp, n * sizeof(T));
      if (error != cudaSuccess) {
          throw std::runtime_error { cudaGetErrorString(error) };
      }
      return tmp;
  }

  auto deallocate(pointer p, size_type n) -> void {
      if (p) {
          auto error = cudaFreeHost(p);
          if (error != cudaSuccess) {
              throw std::runtime_error { cudaGetErrorString(error) };
          }
      }
  }
};

/* Equality operators */
template <class T, class U>
auto operator==(pinned_allocator<T> const &, pinned_allocator<U> const &) -> bool {
    return true;
}

template <class T, class U>
auto operator!=(pinned_allocator<T> const &, pinned_allocator<U> const &) -> bool {
    return false;
}

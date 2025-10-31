#pragma once

#include <vector>
#include <cinttypes>

namespace gpu {

  struct vector {

    explicit vector(uint32_t n);
    vector(const vector & other);
    vector& operator=(const vector & other);
    vector(const std::vector<double> & other);
    vector& operator=(const std::vector<double> & other);
    ~vector();

    uint32_t size() const;

    static void set_memory_pool(uint64_t bytes);

    uint32_t sz;
    double * ptr;

   private:
    void resize(uint32_t new_sz);
  };

  vector zeros(int n);
  double dot(const vector & u, const vector & v);
  double norm(const vector & v);

  vector operator+(const vector & u, const vector & v);
  vector operator-(const vector & u, const vector & v);
  vector operator*(const vector & v, double scale);
  vector operator*(double scale, const vector & v);
  vector operator/(const vector & v, double scale);

}

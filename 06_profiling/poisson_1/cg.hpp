// annotate this file with NVTX ranges
//
// mark the following ranges:
//  -the preamble (the part before the while loop)
//  -the individual iterations
//  -within each iteration, create 3 subranges:
//    - the calculation of `d` up to the assigning new data to `x`
//    - the residual update (r := ... )
//    - updating the search direction (everything after the residual update)

#pragma once

#include <iostream>

template< typename LinearOperatorTypeA, typename VectorType >
VectorType cg(
  const LinearOperatorTypeA & A, 
  const VectorType & b,
  int imax, 
  double epsilon) {

  VectorType x = b * 0.0;
  VectorType r = b;
  VectorType d = r;
  double delta = dot(r, r);
  double delta0 = delta;

  int i = 0;
  while (i < imax && delta > ((epsilon * epsilon) * delta0)) {
    VectorType q = A(d);
    double alpha = delta / dot(d, q);
    x = x + alpha * d;

    r = r - alpha * q;

    double delta_old = delta;
    delta = dot(r, r);

    std::cout << i << " " << delta << std::endl;

    double beta = delta / delta_old;
    d = r + beta * d;
    i++;
  }

  return x;
}

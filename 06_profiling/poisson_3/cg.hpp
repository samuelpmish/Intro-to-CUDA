#pragma once

#include <nvtx3/nvToolsExt.h>

#include <iostream>

template< typename LinearOperatorTypeA, typename VectorType >
VectorType cg(
  const LinearOperatorTypeA & A, 
  const VectorType & b,
  int imax, 
  double epsilon) {

  nvtxRangePushA("preamble");
  VectorType x = b * 0.0;
  VectorType r = b;
  VectorType d = r;
  double delta = dot(r, r);
  double delta0 = delta;
  nvtxRangePop();

  int i = 0;
  while (i < imax && delta > ((epsilon * epsilon) * delta0)) {
    nvtxRangePushA("cg iteration");

    nvtxRangePushA("update solution");
    VectorType q = A(d);
    double alpha = delta / dot(d, q);
    x = x + alpha * d;
    nvtxRangePop();

    nvtxRangePushA("update residual");
    r = r - alpha * q;
    nvtxRangePop();

    nvtxRangePushA("update search direction");
    double delta_old = delta;
    delta = dot(r, r);

    std::cout << i << " " << delta << std::endl;

    double beta = delta / delta_old;
    d = r + beta * d;
    i++;
    nvtxRangePop();

    nvtxRangePop();
  }

  return x;
}


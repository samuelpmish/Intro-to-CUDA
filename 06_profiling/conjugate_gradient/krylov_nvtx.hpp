#pragma once

#include <nvtx3/nvToolsExt.h>

// a direct translation of algorithm B2 from Shewchuk's 
// "An Introduction to the Conjugate Gradient Method
//      Without the Agonizing Pain" edition 1¼
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
    if (i % 50 == 0) {
      r = b - A(x);
    } else {
      r = r - alpha * q;
    }
    nvtxRangePop();

    nvtxRangePushA("update search direction");
    double delta_old = delta;
    delta = dot(r, r);

    double beta = delta / delta_old;
    d = r + beta * d;
    i++;
    nvtxRangePop();

    nvtxRangePop();
  }

  return x;
}

// a direct translation of algorithm B3 from Shewchuk's 
// "An Introduction to the Conjugate Gradient Method
//      Without the Agonizing Pain" edition 1¼
template< typename LinearOperatorTypeM, typename LinearOperatorTypeA, typename VectorType >
VectorType pcg(
  const LinearOperatorTypeM & M, 
  const LinearOperatorTypeA & A, 
  const VectorType & b,
  int imax, 
  double epsilon) {

  VectorType x = b * 0.0;
  VectorType r = b;
  VectorType d = M(r);
  double delta = dot(r, d);
  double delta0 = delta;

  int i = 0;
  while (i < imax && delta > ((epsilon * epsilon) * delta0)) {
    VectorType q = A(d);
    double alpha = delta / dot(d, q);
    x = x + alpha * d;

    if (i % 50 == 0) {
      r = b - A(x);
    } else {
      r = r - alpha * q;
    }

    q = M(r);

    double delta_old = delta;
    delta = dot(r, q);

    double beta = delta / delta_old;
    d = q + beta * d;
    i++;
  }

  return x;
}

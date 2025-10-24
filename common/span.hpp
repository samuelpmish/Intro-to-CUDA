#pragma once

#include "cuda_macros.hpp"

template < int dim, typename T >
struct span;

template < typename T >
struct span< 2, T >{

    static constexpr int dim = 2;

    __host__ __device__ span(T * data, const int (&shape)[dim]) {
        data_ = data;

        shape_[0] = shape[0];
        shape_[1] = shape[1];

        stride_ = shape_[1];
    }

    __host__ __device__ T & operator()(int i, int j) {
        return data_[i * stride_ + j];
    }

    __host__ __device__ const T & operator()(int i, int j) const {
        return data_[i * stride_ + j];
    }

    T * data_;
    int shape_[2];
    int stride_;

};

template < typename T >
struct span< 3, T >{

    static constexpr int dim = 3;

    __host__ __device__ span(T * data, const int (&shape)[dim]) {
        data_ = data;

        shape_[0] = shape[0];
        shape_[1] = shape[1];
        shape_[2] = shape[2];

        stride_[1] = shape_[2];
        stride_[0] = shape_[1] * shape_[2];
    }

    __host__ __device__ T & operator()(int i, int j, int k) {
        return data_[i * stride_[0] + j * stride_[1] + k];
    }

    __host__ __device__ const T & operator()(int i, int j, int k) const {
        return data_[i * stride_[0] + j * stride_[1] + k];
    }

    T * data_;
    int shape_[3];
    int stride_[2];

};

template < typename T >
using span2D = span<2, T>;

template < typename T >
using span3D = span<3, T>;

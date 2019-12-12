//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include <thrust/device_vector.h>
#include <vector>
#include <cublas_v2.h>

namespace cudaqs{
class ThreadCtx{
public:
    cudaStream_t stream;
    cublasHandle_t handle;
};

template<class T>
class Tensor
{
public:
    // Constructors
    // Tensor(const cudaStream_t &stream_input);
    // Tensor(const std::vector<int> &size_input,const cudaStream_t &stream_input);
    Tensor(const ThreadCtx *ctx_ptr_input);
    Tensor(const std::vector<int> &size_input,const ThreadCtx *ctx_ptr_input);
    Tensor(const Tensor<T> &other);

    void copy(const Tensor<T> &other);
    void set_zero();

    void resize(const std::vector<int> &size);
    T* data() const {return (T*)thrust::raw_pointer_cast(v.data());}
    T* data(){return (T*)thrust::raw_pointer_cast(v.data());}
    int numel(const std::vector<int> &size) const;
    int numel() const {return numel(size);};

    // (only work for double type)
    T sum() const;
    void sum_col(Tensor<T> &result) const;
    double norm() const;
    T inner(const Tensor<T> &other) const;
    T max() const;

    Tensor<T>& operator=(const Tensor<T> &other);
    Tensor<T>& operator*=(T s);
    Tensor<T>& operator*=(const Tensor<T> &other);
    Tensor<T>& operator+=(const Tensor<T> &other);
    void negate(Tensor<T> &result) const;
    //Tensor<T> operator-() const;
    void subtract(const Tensor<T> &other, Tensor<T> &result) const;
    //Tensor<T> operator-(const Tensor<T> &other) const;
    void add(const Tensor<T> &other, Tensor<T> &result) const;
    //Tensor<T> operator+(const Tensor<T> &other) const;
    void multiply(T s, Tensor<T> &result) const;
    //Tensor<T> operator*(T s) const;

    void set_stream(cudaStream_t stream_input){stream=stream_input;}

    // Attributes
    std::vector<int> size;
    thrust::device_vector<T> v;
    const ThreadCtx *ctx_ptr;
    cudaStream_t stream;
    cublasHandle_t handle;
};

}

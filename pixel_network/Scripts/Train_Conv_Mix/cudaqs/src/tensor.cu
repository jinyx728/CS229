//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "tensor.h"
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/host_vector.h>
#include <cmath>

using namespace cudaqs;
using thrust::cuda::par;

template<class T>
Tensor<T>::Tensor(const ThreadCtx *ctx_ptr_input):ctx_ptr(ctx_ptr_input)
{
    if(ctx_ptr!=nullptr){
        stream=ctx_ptr_input->stream;
        handle=ctx_ptr_input->handle;
    }
    else{
        stream=0;
    }
}

template<class T>
Tensor<T>::Tensor(const std::vector<int> &size_input,const ThreadCtx *ctx_ptr_input):size(size_input),ctx_ptr(ctx_ptr_input)
{
    if(ctx_ptr!=nullptr){
        stream=ctx_ptr_input->stream;
        handle=ctx_ptr_input->handle;
    }
    else{
        stream=0;
    }
    v.resize(numel());set_zero();
}


template<class T>
Tensor<T>::Tensor(const Tensor<T> &other):ctx_ptr(other.ctx_ptr),stream(other.stream),handle(other.handle)
{
    *this=other;
}

template<class T>
T Tensor<T>::sum() const
{
    cublasSetStream(handle,stream);
    double result=0;
    cublasDasum(handle,numel(),(double*)data(),1,&result);
    return (T)result;
    // return thrust::reduce(par.on(stream),v.begin(),v.end(),(T)0,thrust::plus<T>());
}

template<class T>
__global__ void sum_col_kernel(T* in,int nrows,int ncols,T* out)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=ncols)
        return;
    T col_sum=0;
    for(int j=0;j<nrows;j++){
        col_sum+=in[j*ncols+i];
    }
    out[i]=col_sum;
}


template<class T>
void Tensor<T>::sum_col(Tensor<T> &result) const
{
    int nrows=size[0],ncols=size[1];
    const int block_size=256;
    const int num_blocks=(ncols+block_size-1)/block_size;
    sum_col_kernel<<<num_blocks,block_size,0,stream>>>(data(),nrows,ncols,result.data());
}

template<class T>
int Tensor<T>::numel(const std::vector<int> &size) const
{
    int t=1;for(auto i:size) t*=i;
    return t;
}

template<class T>
void Tensor<T>::resize(const std::vector<int> &size)
{
    v.resize(numel(size));
    this->size=size;
}

template<class T>
double Tensor<T>::norm() const
{
    // auto square=[]__device__(const T &x){return x*x;};
    // T sum_sqr=thrust::transform_reduce(par.on(stream),v.begin(),v.end(),square,(T)0,thrust::plus<T>());
    // return sqrt(sum_sqr);
    cublasSetStream(handle,stream);
    double result=0;
    cublasDnrm2(handle,numel(),(double*)data(),1,&result);
    return (T)result;
}

template<class T>
T Tensor<T>::inner(const Tensor<T> &other) const
{
    cublasSetStream(handle,stream);
    double result=0;
    cublasDdot(handle,numel(),(double*)data(),1,(double*)other.data(),1,&result);
    return (T)result;
    // return thrust::inner_product(par.on(stream),v.begin(),v.end(),other.v.begin(),(T)0);
}

template<class T>
T Tensor<T>::max() const
{
    cublasSetStream(handle,stream);
    int id=0;
    cublasIdamax(handle,numel(),(double*)data(),1,&id);
    double result=0;
    cudaMemcpyAsync(&result,data()+id-1,sizeof(double),cudaMemcpyDeviceToHost,stream);
    // cudaStreamSynchronize(stream);
    // printf("max:%f\n",result);
    return (T)result;
    // T result=*thrust::max_element(par.on(stream),v.begin(),v.end());
    // printf("max:%f\n",(double)result);
    // return result;
}


template<class T>
void Tensor<T>::copy(const Tensor<T> &other)
{
    // stream=other.stream; // do I want this?
    assert(this->numel() == other.numel());
    cudaMemcpyAsync(data(),other.data(),numel()*sizeof(T),cudaMemcpyDeviceToDevice,stream);
    // thrust::copy(par.on(stream),other.v.begin(),other.v.end(),v.begin());
}

template<class T>
void Tensor<T>::set_zero()
{
    // thrust::fill(par.on(stream),v.begin(),v.end(),(T)0);
    cudaMemsetAsync(data(),0,numel()*sizeof(T),stream);
}

template<class T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T> &other)
{
    resize(other.size);
    copy(other);
    return *this;
}

template<class T,typename Func>
__global__ void binary_kernel(const T* a,const T* b,T *out,int n,Func f)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)
        return;
    out[i]=f(a[i],b[i]);
}

template<class T,typename Func>
void binary_op(const T* a,const T* b,T *out,int n,Func f,const cudaStream_t stream)
{
    const int block_size=256;
    const int num_blocks=(n+block_size-1)/block_size;
    binary_kernel<<<num_blocks,block_size,0,stream>>>(a,b,out,n,f);
}

template<class T,typename Func>
__global__ void unary_kernel(const T* a,T *out,int n,Func f)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)
        return;
    out[i]=f(a[i]);
}

template<class T,typename Func>
void unary_op(const T* a,T *out,int n,Func f,const cudaStream_t stream)
{
    const int block_size=256;
    const int num_blocks=(n+block_size-1)/block_size;
    unary_kernel<<<num_blocks,block_size,0,stream>>>(a,out,n,f);
}

template<class T>
Tensor<T>& Tensor<T>::operator*=(T s)
{
    // auto mul=[=]__device__(T x){return x*s;};
    // thrust::transform(par.on(stream),v.begin(),v.end(),v.begin(),mul);
    auto f=[=]__device__(T x){return s*x;};
    unary_op(data(),data(),numel(),f,stream);
    return *this;
}

template<class T>
__global__ void mul_broadcast(T* a,T* b,T* out,int size0,int size1)
{
    // a: D*N, b: 1*N, out: D*N, size0: D, size1: N
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=size0*size1)
        return;
    int i0=i%size1;
    out[i]=a[i]*b[i0];
}

template<class T>
Tensor<T>& Tensor<T>::operator*=(const Tensor<T> &other)
{
    // assert(stream!=0);
    int n=numel();
    const int block_size=256;
    const int num_blocks=(n+block_size-1)/block_size;
    if(other.size[0]==1){
        mul_broadcast<<<num_blocks,block_size,0,stream>>>(data(),other.data(),data(),size[0],size[1]);
    }
    else{
        auto f=[]__device__(T a,T b){return a*b;};
        binary_op(data(),other.data(),data(),n,f,stream);
        // thrust::transform(par.on(stream),v.begin(),v.end(),other.v.begin(),v.begin(),thrust::multiplies<T>());
    }
    return *this;
}

template<class T>
Tensor<T>& Tensor<T>::operator+=(const Tensor<T> &other)
{
    // thrust::transform(par.on(stream),v.begin(),v.end(),other.v.begin(),v.begin(),thrust::plus<T>());
    auto f=[]__device__(T a,T b){return a+b;};
    binary_op(data(),other.data(),data(),numel(),f,stream);
    return *this;
}

template<class T>
void Tensor<T>::negate(Tensor<T> &result) const
{
    // thrust::transform(par.on(stream),v.begin(),v.end(),result.v.begin(),thrust::negate<T>());
    auto f=[]__device__(T x){return -x;};
    unary_op(data(),result.data(),numel(),f,stream);
}

/*
template<class T>
Tensor<T> Tensor<T>::operator-() const
{
    Tensor<T> result(size,stream);
    thrust::transform(par.on(stream),v.begin(),v.end(),result.v.begin(),thrust::negate<T>());
    return result; 
}
*/

template<class T>
void Tensor<T>::add(const Tensor<T> &other, Tensor<T> &result) const
{
    // thrust::transform(par.on(stream),v.begin(),v.end(),other.v.begin(),result.v.begin(),thrust::plus<T>());
    auto f=[]__device__(T a,T b){return a+b;};
    binary_op(data(),other.data(),result.data(),numel(),f,stream);
}

/*
template<class T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> &other) const
{
    Tensor<T> result(size,stream);
    thrust::transform(par.on(stream),v.begin(),v.end(),other.v.begin(),result.v.begin(),thrust::plus<T>());
    return result;
}
*/

template<class T>
void Tensor<T>::subtract(const Tensor<T> &other, Tensor<T> &result) const
{
    // thrust::transform(par.on(stream),v.begin(),v.end(),other.v.begin(),result.v.begin(),thrust::minus<T>());
    auto f=[]__device__(T a,T b){return a-b;};
    binary_op(data(),other.data(),result.data(),numel(),f,stream);
}

/*
template<class T>
Tensor<T> Tensor<T>::operator-(const Tensor<T> &other) const
{
    Tensor<T> result(size,stream);
    thrust::transform(par.on(stream),v.begin(),v.end(),other.v.begin(),result.v.begin(),thrust::minus<T>());
    return result;
}
*/

template<class T>
void Tensor<T>::multiply(T s, Tensor<T> &result) const
{
    // auto mul=[=]__device__(T x){return x*s;};
    // thrust::transform(par.on(stream),v.begin(),v.end(),result.v.begin(),mul);
    auto f=[=]__device__(T a){return a*s;};
    unary_op(data(),result.data(),numel(),f,stream);
}

/*
template<class T>
Tensor<T> Tensor<T>::operator*(T s) const
{
    auto mul=[=]__device__(T x){return x*s;};
    Tensor<T> result(size,stream);
    thrust::transform(par.on(stream),v.begin(),v.end(),result.v.begin(),mul);
    return result;
}
*/

template class Tensor<double>;
template class Tensor<int>;


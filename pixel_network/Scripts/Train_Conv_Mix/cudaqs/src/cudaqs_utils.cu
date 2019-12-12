//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "cudaqs_utils.h"
#include <cuda.h>

using namespace cudaqs;

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


template<class T>
__global__ void copy_kernel(const T* __restrict__ in,T* __restrict__ out,int n)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)
        return;
    out[i]=in[i];
}

template<class T>
void cudaqs::from_torch_tensor(const at::Tensor &in,Tensor<T> &out,bool new_tensor)
{
    std::vector<int> size;
    for(auto s:in.sizes()) size.push_back((int)s);
    if(new_tensor) out.resize(size);

    at::Tensor flat_in=in.view(-1).contiguous();
    int numel=in.numel();

    CHECK_INPUT(flat_in);
    // const int block_size=256;
    // int num_blocks=(numel+block_size-1)/block_size;
    // copy_kernel<T><<<num_blocks,block_size,0,out.stream>>>(flat_in.data<T>(),out.data(),numel);
    cudaMemcpyAsync(out.data(),flat_in.data<T>(),numel*sizeof(T),cudaMemcpyDeviceToDevice,out.stream);
}

template<class T>
void cudaqs::to_torch_tensor(const Tensor<T> &in,at::Tensor &out,bool new_tensor)
{
    // typedef typename at::IntArrayRef::value_type value_type;
    typedef int64_t value_type;
    std::vector<value_type> size;
    for(auto s:in.size) size.push_back(s);

    if(new_tensor){
        auto options=torch::TensorOptions().dtype(out.dtype()).device(out.device());
        out=torch::zeros({in.numel()},options);
    }
    else{
        out=out.view(-1);
    }
    CHECK_INPUT(out);

    int numel=in.numel();
    // const int block_size=256;
    // int num_blocks=(numel+block_size-1)/block_size;
    // copy_kernel<T><<<num_blocks,block_size,0,in.stream>>>(in.data(),out.data<T>(),numel);
    cudaMemcpyAsync(out.data<T>(),in.data(),numel*sizeof(T),cudaMemcpyDeviceToDevice,in.stream);
    out=out.view(size);
}

template void cudaqs::from_torch_tensor<int>(const at::Tensor &in,Tensor<int> &out,bool new_tensor);
template void cudaqs::from_torch_tensor<double>(const at::Tensor &in,Tensor<double> &out,bool new_tensor);
template void cudaqs::to_torch_tensor<int>(const Tensor<int> &in,at::Tensor &out,bool new_tensor);
template void cudaqs::to_torch_tensor<double>(const Tensor<double> &in,at::Tensor &out,bool new_tensor);

//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "cudaqs.h"
#include <torch/extension.h>
#include "../include/ctpl/ctpl_stl.h"
#include <cuda.h>

__device__
int switch_2d_to_1d_kernel(int i, int j, int N){
    return i*N + j;
}

__global__
void row_sum_cuda_kernel(int N, double *row_sums, torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> input_a){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i+=stride){
        float sum = 0;
        for(int j=0; j<N; j++){
            sum += input_a[i][j];
        }
        row_sums[i] = sum;
    }
}

std::vector<double> row_sum_cuda(torch::Tensor input){
    int N = input.size(0);
    double *row_sums;

    cudaMallocManaged(&row_sums, N*sizeof(double));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    auto input_a = input.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    row_sum_cuda_kernel<<<numBlocks, blockSize>>>(N, row_sums, input_a);
  
    cudaDeviceSynchronize();

    std::vector<double> row_sums_vec = std::vector<double>(row_sums, row_sums + N);
    cudaFree(row_sums);

    return row_sums_vec;
}

__global__
void row_sum_cuda_ctpl_kernel(int N, torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> input_a, int sample_id, float* row_sums){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = index; j < N; j+=stride){
        atomicAdd(&row_sums[sample_id], input_a[sample_id][j]);
    }
}

std::vector<float> row_sum_cuda_ctpl(torch::Tensor input, int num_ctpl_threads){
    int N = input.size(0);
    std::shared_ptr<ctpl::thread_pool> pool=nullptr;
    pool=std::shared_ptr<ctpl::thread_pool>(new ctpl::thread_pool(num_ctpl_threads));
    std::vector<std::future<void>> futures(N);

    float* row_sums;
    cudaMallocManaged(&row_sums, N*sizeof(float));
    auto input_a = input.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();

    auto process_sample=[&](int thread_id,int sample_id, at::PackedTensorAccessor<float, 2UL, at::RestrictPtrTraits, size_t> input_a, float* row_sums){
        int blockSize = 256;
        int numBlocks = 3;
        row_sum_cuda_ctpl_kernel<<<numBlocks, blockSize>>>(N, input_a, sample_id, row_sums);
    };
    for(int sample_id=0;sample_id<N;sample_id++){
        futures[sample_id]=pool->push(process_sample,sample_id,input_a,row_sums);
    }
    for(int sample_id=0;sample_id<N;sample_id++){
        futures[sample_id].get();
    }

    cudaDeviceSynchronize();

    std::vector<float> row_sums_vec = std::vector<float>(row_sums, row_sums + N);

    cudaFree(row_sums);

    return row_sums_vec;
}

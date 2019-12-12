//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "cudaqs_global.h"
#include "cudaqs_utils.h"
#include <chrono>
#include "cuda_profiler_api.h"
using namespace cudaqs;

std::shared_ptr<ctpl::thread_pool> cudaqs::pool;
int cudaqs::batch_size;
bool cudaqs::use_multi_thread;
bool cudaqs::use_cuda_stream;
bool cudaqs::verbose;
std::vector<ThreadCtx> cudaqs::thread_ctxs;

// SpringData cudaqs::spring_data;
// AxialData cudaqs::axial_data;
// SpringSystem cudaqs::spring_system;
// NewtonOpt cudaqs::forward_opt;

void cudaqs::init(int n_vts,int batch_size_,bool use_multi_thread_,bool verbose_)
{
    batch_size=batch_size_;
    use_multi_thread=use_multi_thread_;
    use_cuda_stream=true;
    verbose=verbose_;
    pool=std::shared_ptr<ctpl::thread_pool>(new ctpl::thread_pool(batch_size_));
    if(use_multi_thread&&use_cuda_stream){
        thread_ctxs.resize(batch_size);
        for(int i=0;i<batch_size;i++){
            cudaStreamCreateWithFlags(&thread_ctxs[i].stream,cudaStreamNonBlocking);
            cublasCreate(&thread_ctxs[i].handle);
        }
    }
    else{
        thread_ctxs.resize(1);
        thread_ctxs[0].stream=0;
        cublasCreate(&thread_ctxs[0].handle);
    }

    printf("init:use_multi_thread:%d\n",use_multi_thread);
}

SpringData cudaqs::init_spring(const at::Tensor &edges,const at::Tensor &l0,const at::Tensor &k)
{   
    SpringData spring_data;
    auto options=torch::TensorOptions().dtype(torch::kInt32);
    at::Tensor edges_i32=edges.to(options);
    int n_edges=edges_i32.size(0);
    spring_data.n_edges=n_edges;
    edges_i32=edges_i32.permute({1,0}).contiguous();
    from_torch_tensor<int>(edges_i32,spring_data.edges);
    from_torch_tensor<double>(l0,spring_data.l0);
    from_torch_tensor<double>(k,spring_data.k);
    return spring_data;
}

AxialData cudaqs::init_axial(const at::Tensor &i,const at::Tensor &w,const at::Tensor &k)
{
    AxialData axial_data;
    auto options=torch::TensorOptions().dtype(torch::kInt32);
    at::Tensor i_i32=i.to(options);
    int n_edges=i_i32.size(0);
    axial_data.n_edges=n_edges;
    at::Tensor i_t=i_i32.permute({1,0}).contiguous();
    from_torch_tensor<int>(i_t,axial_data.i);
    at::Tensor w_t=w.permute({1,0}).contiguous();
    from_torch_tensor<double>(w_t,axial_data.w);
    at::Tensor k_t=k.contiguous();
    from_torch_tensor<double>(k_t,axial_data.k);
    return axial_data;
}

SpringSystem cudaqs::init_system(const int n_vts,const SpringData &spring_data,const AxialData &axial_data)
{
    SpringSystem system;system.init(n_vts,spring_data,axial_data);
    return system;
}

NewtonOpt cudaqs::init_forward(const SpringSystem &system)
{
    NewtonOpt opt;opt.init(&system);
    return opt;
}

std::vector<OptDataPtr> cudaqs::init_opt_data(int batch_size,int n_vts,int n_edges)
{
    std::vector<OptDataPtr> opt_datas;
    for(int sample_id=0;sample_id<batch_size;sample_id++){
        ThreadCtx *ctx_ptr=use_multi_thread?&thread_ctxs[sample_id]:&thread_ctxs[0];
        opt_datas.push_back(std::make_shared<OptData>(ctx_ptr));
        opt_datas[sample_id]->resize({3,n_vts},n_edges);
    }
    return opt_datas;
}

std::vector<at::Tensor> cudaqs::solve_forward(const NewtonOpt &forward_opt,const at::Tensor &anchor_in,const at::Tensor &stiffen_anchor_in,std::vector<OptDataPtr> &opt_datas)
{
    int n_samples=anchor_in.size(0);
    at::Tensor anchor_t=anchor_in.permute({0,2,1}).contiguous();
    int d=anchor_t.size(1),n_vts=anchor_t.size(2);
    auto x_options=torch::TensorOptions().device(anchor_in.device()).dtype(anchor_in.dtype());
    at::Tensor out_x=torch::zeros(anchor_t.sizes(),x_options);
    auto success_options=torch::TensorOptions().dtype(torch::kInt32);
    at::Tensor success=torch::zeros(n_samples,success_options);

    auto process_sample=[&](int id,int sample_id){
        OptDataPtr opt_data=opt_datas[sample_id];
        Tensor<double> &anchor=opt_data->forward_data.anchor;from_torch_tensor(anchor_t[sample_id],anchor,false);
        Tensor<double> &stiffen_anchor=opt_data->forward_data.stiffen_anchor;from_torch_tensor(stiffen_anchor_in[sample_id],stiffen_anchor,false);

        std::chrono::system_clock::time_point start,end;
        if(verbose){
            start=std::chrono::system_clock::now();
        }
        Tensor<double> &xi=opt_data->forward_data.x;
        success[sample_id]=forward_opt.solve(anchor,stiffen_anchor,*opt_data,xi);
        at::Tensor xt=out_x[sample_id];
        to_torch_tensor(xi,xt,false);
        
        if(verbose){
            end=std::chrono::system_clock::now();
            double time=std::chrono::duration<double>(end-start).count();
            printf("solve_forward:%d,%fs\n",sample_id,time);
        }
    };

    if(use_multi_thread){
        // cudaProfilerStart();
        std::vector<std::future<void> > futures(n_samples);
        for(int sample_id=0;sample_id<n_samples;sample_id++){
            futures[sample_id]=pool->push(process_sample,sample_id);
        }
        for(int sample_id=0;sample_id<n_samples;sample_id++){
            futures[sample_id].get();
        }
        // cudaProfilerStop();
    }
    else{
        for(int sample_id=0;sample_id<n_samples;sample_id++){
            process_sample(sample_id,sample_id);
        }
    }
    // return {torch::cat(x_vec,0),success};
    out_x=out_x.permute({0,2,1}).contiguous();
    return {out_x,success};
}
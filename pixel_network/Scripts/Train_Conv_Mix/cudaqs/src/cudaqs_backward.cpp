//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "cudaqs_global.h"
#include "cudaqs_utils.h"
#include <chrono>
using namespace cudaqs;

BackwardOpt cudaqs::init_backward(const SpringSystem &system,bool use_variable_stiffen_anchor)
{
    BackwardOpt opt;opt.init(&system,use_variable_stiffen_anchor);
    return opt;
}

std::vector<at::Tensor> cudaqs::solve_backward(const BackwardOpt &backward_opt,const at::Tensor &dl_in,const at::Tensor &x_in,const at::Tensor &anchor_in,const at::Tensor &stiffen_anchor_in,std::vector<OptDataPtr> &opt_datas)
{
    int n_samples=anchor_in.size(0);
    int n_vts=anchor_in.size(1);
    at::Tensor dl_t=dl_in.permute({0,2,1}).contiguous();
    at::Tensor x_t=x_in.permute({0,2,1}).contiguous();
    at::Tensor anchor_t=anchor_in.permute({0,2,1}).contiguous();
    auto x_options=torch::TensorOptions().device(anchor_in.device()).dtype(anchor_in.dtype());
    at::Tensor out_da=torch::zeros(anchor_t.sizes(),x_options);
    at::Tensor out_dstiffen_anchor=torch::zeros({n_samples,n_vts},x_options);
    auto success_options=torch::TensorOptions().dtype(torch::kInt32);
    at::Tensor success=torch::zeros(n_samples,success_options);

    auto process_sample=[&](int id,int sample_id){
        OptDataPtr opt_data=opt_datas[sample_id];
        Tensor<double> &dl=opt_data->backward_data.dl;from_torch_tensor(dl_t[sample_id],dl);
        Tensor<double> &x=opt_data->forward_data.x;from_torch_tensor(x_t[sample_id],x);
        Tensor<double> &anchor=opt_data->forward_data.anchor;from_torch_tensor(anchor_t[sample_id],anchor);
        Tensor<double> &stiffen_anchor=opt_data->forward_data.stiffen_anchor;from_torch_tensor(stiffen_anchor_in[sample_id],stiffen_anchor);

        std::chrono::system_clock::time_point start,end;
        if(verbose){
            start=std::chrono::system_clock::now();
        }
        Tensor<double> &da_i=opt_data->backward_data.da;
        Tensor<double> &dstiffen_anchor_i=opt_data->backward_data.dstiffen_anchor;
        success[sample_id]=backward_opt.solve(dl,x,anchor,stiffen_anchor,*opt_data,da_i,dstiffen_anchor_i);
        at::Tensor da_t=out_da[sample_id];
        to_torch_tensor(da_i,da_t,false);
        if(backward_opt.use_variable_stiffen_anchor){
            at::Tensor dstiffen_anchor_t=out_dstiffen_anchor[sample_id];
            to_torch_tensor(dstiffen_anchor_i,dstiffen_anchor_t,false);
        }
        
        if(verbose){
            end=std::chrono::system_clock::now();
            double time=std::chrono::duration<double>(end-start).count();
            printf("solve_backward:%d,%fs\n",sample_id,time);
        }
    };

    if(use_multi_thread){
        std::vector<std::future<void> > futures(n_samples);
        for(int sample_id=0;sample_id<n_samples;sample_id++){
            futures[sample_id]=pool->push(process_sample,sample_id);
        }
        for(int sample_id=0;sample_id<n_samples;sample_id++){
            futures[sample_id].get();
        }
    }
    else{
        for(int sample_id=0;sample_id<n_samples;sample_id++){
            process_sample(sample_id,sample_id);
        }
    }

    out_da=out_da.permute({0,2,1}).contiguous();
    if(backward_opt.use_variable_stiffen_anchor){
        out_dstiffen_anchor=out_dstiffen_anchor.contiguous();
    }
    return {out_da,out_dstiffen_anchor,success};
}
//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "cvx_opt_global.h"
#include <algorithm>
std::vector<BackwardOpt> backward_opts;

void init_backward()
{
    std::vector<BackwardOpt> &opts=backward_opts;
    opts.resize(batch_size);
    for(int i=0;i<batch_size;i++){
        opts[i].init_solver(&forward_opts[i]);
    }
}

void get_sol(const std::vector<at::Tensor> &tensors,int sample_id,Solution &s)
{
    tensor_to_vector(tensors[0][sample_id],s.x);
    tensor_to_vector(tensors[1][sample_id],s.y);
    tensor_to_vector(tensors[2][sample_id],s.z);
    tensor_to_vector(tensors[3][sample_id],s.s);
    s.success=tensors[4].accessor<idxint,1>()[sample_id];
}

at::Tensor solve_backward(at::Tensor grad_t,at::Tensor tgt_x_t,std::vector<at::Tensor> tensors)
{
    std::vector<BackwardOpt> &opts=backward_opts;
    int n_samples=tgt_x_t.size(0);
    std::vector<std::future<void> > futures(n_samples);
    std::vector<at::Tensor> out_grads(n_samples);
    std::vector<Solution> sols(n_samples);init_sols(sols);
    std::vector<std::vector<pfloat> > grad_vs(n_samples),tgt_x_vs(n_samples);
    for(int i=0;i<n_samples;i++){
        grad_vs[i].resize(opts[i].n_vts*D);tgt_x_vs[i].resize(opts[i].n_vts*D);
    }

    auto process_sample=[&](int id,int sample_id){
        if(verbose){
            auto now=std::chrono::system_clock::now();
            printf("backward,sample_id:%d,cpu:%d,start:%f s\n",sample_id,sched_getcpu(),std::chrono::duration<double>(now-init_time).count());
        }

        std::vector<pfloat> &grad_v=grad_vs[sample_id];tensor_to_vector(grad_t[sample_id],grad_v,false);
        std::vector<pfloat> &tgt_x_v=tgt_x_vs[sample_id];tensor_to_vector(tgt_x_t[sample_id],tgt_x_v,false);
        Solution &sol=sols[sample_id];
        get_sol(tensors,sample_id,sol);
        idxint success=sol.success;
        BackwardOpt &opt=opts[sample_id];
        std::vector<pfloat> out_grad_v;

        if(success!=1){
            printf("backward input failed, sample_id:%d\n",sample_id);
        }
        if(success==1){
            opt.solve(tgt_x_v,sol,grad_v,out_grad_v);
            success=!is_unusual_dx(out_grad_v);
        }
        if(success!=1){
            printf("backward failed,use 0 grad,sample_id:%d\n",sample_id);
            out_grad_v.resize(opt.n_vts*D);std::fill(out_grad_v.begin(),out_grad_v.end(),0);
        }

        out_grads[sample_id]=vector_to_tensor(out_grad_v).view({1,-1});
        if(verbose){
            auto now=std::chrono::system_clock::now();
            printf("solve,sample_id:%d,cpu:%d,end:%f s\n",sample_id,sched_getcpu(),std::chrono::duration<double>(now-init_time).count());
        }
    };

    if(use_multi_thread){
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

    if(verbose){
        auto now=std::chrono::system_clock::now();
        printf("solve,complete:,end:%f s\n",std::chrono::duration<double>(now-init_time).count());
    }

    backward_counter+=1;
    auto cat=at::cat(out_grads,0);
    return cat;
}

std::vector<at::Tensor> solve_backward_variable_m(at::Tensor grad_t,at::Tensor tgt_x_t,at::Tensor m_t,std::vector<at::Tensor> tensors)
{
    std::vector<BackwardOpt> &opts=backward_opts;
    int n_samples=tgt_x_t.size(0);
    std::vector<std::future<void> > futures(n_samples);
    std::vector<at::Tensor> out_grads(n_samples),out_m_grads(n_samples);
    std::vector<Solution> sols(n_samples);init_sols(sols);
    std::vector<std::vector<pfloat> > grad_vs(n_samples),tgt_x_vs(n_samples),m_vs(n_samples);
    for(int i=0;i<n_samples;i++){
        grad_vs[i].resize(opts[i].n_vts*D);tgt_x_vs[i].resize(opts[i].n_vts*D);m_vs[i].resize(opts[i].n_vts);
    }

    auto process_sample=[&](int id,int sample_id){
        if(verbose){
            auto now=std::chrono::system_clock::now();
            printf("backward,sample_id:%d,cpu:%d,start:%f s\n",sample_id,sched_getcpu(),std::chrono::duration<double>(now-init_time).count());
        }

        std::vector<pfloat> &grad_v=grad_vs[sample_id];tensor_to_vector(grad_t[sample_id],grad_v,false);
        std::vector<pfloat> &tgt_x_v=tgt_x_vs[sample_id];tensor_to_vector(tgt_x_t[sample_id],tgt_x_v,false);
        std::vector<pfloat> &m_v=m_vs[sample_id];tensor_to_vector(m_t[sample_id],m_v,false);
        Solution &sol=sols[sample_id];
        get_sol(tensors,sample_id,sol);
        idxint success=sol.success;
        BackwardOpt &opt=opts[sample_id];
        std::vector<pfloat> out_grad_v;
        std::vector<pfloat> out_grad_m_v;

        if(success!=1){
            printf("backward input failed, sample_id:%d\n",sample_id);
        }
        if(success==1){
            opt.solve(tgt_x_v,m_v,sol,grad_v,out_grad_v,out_grad_m_v);
            success=!is_unusual_x(out_grad_v);
        }
        if(success!=1){
            printf("backward failed,use 0 grad,sample_id:%d\n",sample_id);
            out_grad_v.resize(opt.n_vts*D);std::fill(out_grad_v.begin(),out_grad_v.end(),0);
            out_grad_m_v.resize(opt.n_vts);std::fill(out_grad_m_v.begin(),out_grad_m_v.end(),0);
        }

        out_grads[sample_id]=vector_to_tensor(out_grad_v).view({1,-1});
        out_m_grads[sample_id]=vector_to_tensor(out_grad_m_v).view({1,-1});
        if(verbose){
            auto now=std::chrono::system_clock::now();
            printf("solve,sample_id:%d,cpu:%d,end:%f s\n",sample_id,sched_getcpu(),std::chrono::duration<double>(now-init_time).count());
        }
    };

    if(use_multi_thread){
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

    if(verbose){
        auto now=std::chrono::system_clock::now();
        printf("solve,complete:,end:%f s\n",std::chrono::duration<double>(now-init_time).count());
    }

    backward_counter+=1;
    return {at::cat(out_grads,0),at::cat(out_m_grads,0)};
}

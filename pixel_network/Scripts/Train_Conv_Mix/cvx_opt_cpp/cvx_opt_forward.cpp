//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "cvx_opt_global.h"
#include <cstdio>
#include <algorithm>
std::vector<ForwardOpt> forward_opts;
PreinitData pre_data;
 
void init_forward(at::Tensor m_t,at::Tensor edges_t,at::Tensor l0_t)
{
    std::vector<ForwardOpt> &opts=forward_opts;
    std::vector<pfloat> m_v;
    tensor_to_vector(m_t,m_v);
    std::vector<idxint> edges_v;
    tensor_to_vector(edges_t,edges_v);
    std::vector<pfloat> l0_v;
    tensor_to_vector(l0_t,l0_v);
    int n_vts=m_t.size(0);
    int n_edges=edges_t.size(0);
    opts.resize(batch_size);
    for(int i=0;i<batch_size;i++){
        opts[i].init_solver(m_v,edges_v,l0_v,n_vts,n_edges,&pre_data);
    }
}
 
void init_lap(at::Tensor Lpr_t,at::Tensor Ljc_t,at::Tensor Lir_t,double lmd_lap)
{
    pre_data.use_lap=true;
    tensor_to_vector(Lpr_t,pre_data.Lpr);
    tensor_to_vector(Ljc_t,pre_data.Ljc);
    tensor_to_vector(Lir_t,pre_data.Lir);
    pre_data.lmd_lap=lmd_lap;
}

void init_spring(at::Tensor youngs_modulus,double lmd_k)
{
    pre_data.use_spring=true;pre_data.lmd_k=lmd_k;
    tensor_to_vector(youngs_modulus,pre_data.youngs_modulus);
}

void init_options(int max_it,pfloat feastol,pfloat abstol,pfloat reltol,pfloat feastol_inacc,pfloat abstol_inacc,pfloat reltol_inacc)
{
    std::vector<ForwardOpt> &opts=forward_opts;
    for(int i=0;i<batch_size;i++){
        opts[i].init_options(max_it,feastol,abstol,reltol,feastol_inacc,abstol_inacc,reltol_inacc);
    }
}

std::vector<at::Tensor> merge_solutions(const std::vector<Solution> &sols)
{
    std::vector<std::vector<at::Tensor> > tensor_lists(4);
    std::vector<idxint> success(sols.size());
    for(uint sample_id=0;sample_id<sols.size();sample_id++){
        const Solution &sol=sols[sample_id];
        tensor_lists[0].push_back(vector_to_tensor(sol.x).view({1,-1}));
        tensor_lists[1].push_back(vector_to_tensor(sol.y).view({1,-1}));
        tensor_lists[2].push_back(vector_to_tensor(sol.z).view({1,-1}));
        tensor_lists[3].push_back(vector_to_tensor(sol.s).view({1,-1}));
        success[sample_id]=sol.success;
    }
    std::vector<at::Tensor> ret(tensor_lists.size()+1);
    for(uint i=0;i<tensor_lists.size();i++){
        ret[i]=at::cat(tensor_lists[i],0);
    }
    ret[tensor_lists.size()]=vector_to_tensor(success);
    return ret;
}

void init_sols(std::vector<Solution> &sols)
{
    std::vector<ForwardOpt> &opts=forward_opts;
    for(uint i=0;i<sols.size();i++){
        sols[i].x.resize(opts[i].n);
        sols[i].y.resize(opts[i].p);
        sols[i].z.resize(opts[i].m);
        sols[i].s.resize(opts[i].m);
        sols[i].success=0;
    }
}

std::vector<at::Tensor> solve_forward(at::Tensor tgt_x_t)
{
    std::vector<ForwardOpt> &opts=forward_opts;
    int n_samples=tgt_x_t.size(0);
    std::vector<Solution> sols(n_samples); 
    init_sols(sols);
    std::vector<std::future<void> > futures(n_samples);
    std::vector<std::vector<pfloat> > tgt_x_vs(n_samples);
    for(int i=0;i<n_samples;i++){
        tgt_x_vs[i].resize(opts[i].n_vts*D);
    }

    auto process_sample=[&](int id,int sample_id){
        if(verbose){
            auto now=std::chrono::system_clock::now();
            printf("solve,sample_id:%d,cpu:%d,start:%f s\n",sample_id,sched_getcpu(),std::chrono::duration<double>(now-init_time).count());
        }
        at::Tensor tgt_x_ti=tgt_x_t[sample_id];

        std::vector<pfloat> &tgt_x_v=tgt_x_vs[sample_id];
        tensor_to_vector(tgt_x_ti,tgt_x_v,false);

        ForwardOpt &opt=opts[sample_id];
        int exitflag=opt.solve(tgt_x_v,sols[sample_id],verbose);
        print_exit_flag_if_ncsry(exitflag,sample_id,verbose);
        if(sols[sample_id].success!=1){
            printf("forward failed,sample_id:%d\n",sample_id);
        }
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
    forward_counter+=1;
    if(verbose){
        auto now=std::chrono::system_clock::now();
        printf("solve,complete:,end:%f s\n",std::chrono::duration<double>(now-init_time).count());
    }

    return merge_solutions(sols);
}

std::vector<at::Tensor> solve_forward_variable_m(at::Tensor tgt_x_t,at::Tensor m_t)
{
    std::vector<ForwardOpt> &opts=forward_opts;
    int n_samples=tgt_x_t.size(0);
    std::vector<Solution> sols(n_samples); 
    init_sols(sols);
    std::vector<std::future<void> > futures(n_samples);
    std::vector<std::vector<pfloat> > tgt_x_vs(n_samples);
    std::vector<std::vector<pfloat> > m_vs(n_samples);
    for(int i=0;i<n_samples;i++){
        tgt_x_vs[i].resize(opts[i].n_vts*D);
        m_vs[i].resize(opts[i].n_vts);
    }

    auto process_sample=[&](int id,int sample_id){
        if(verbose){
            auto now=std::chrono::system_clock::now();
            printf("solve,sample_id:%d,cpu:%d,start:%f s\n",sample_id,sched_getcpu(),std::chrono::duration<double>(now-init_time).count());
        }
        at::Tensor tgt_x_ti=tgt_x_t[sample_id];
        std::vector<pfloat> &tgt_x_v=tgt_x_vs[sample_id];
        tensor_to_vector(tgt_x_ti,tgt_x_v,false);

        at::Tensor m_ti=m_t[sample_id];
        std::vector<pfloat> &m_v=m_vs[sample_id];
        tensor_to_vector(m_ti,m_v,false);

        ForwardOpt &opt=opts[sample_id];
        int exitflag=opt.solve(tgt_x_v,m_v,sols[sample_id],verbose);
        print_exit_flag_if_ncsry(exitflag,sample_id,verbose);
        if(sols[sample_id].success!=1){
            printf("forward failed,sample_id:%d\n",sample_id);
        }
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
    forward_counter+=1;
    if(verbose){
        auto now=std::chrono::system_clock::now();
        printf("solve,complete:,end:%f s\n",std::chrono::duration<double>(now-init_time).count());
    }

    return merge_solutions(sols);
}
//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "ecos_opt.h"
#include "ctpl_stl.h"

// void print(at::Tensor z)
// {
// 	std::cout<<z<<std::endl;
// }
#define D 3
std::vector<EcosOpt> opts;
std::shared_ptr<ctpl::thread_pool> pool=nullptr;
std::chrono::system_clock::time_point init_time;

template<class T>
void tensor_to_vector(at::Tensor t,std::vector<T> &v)
{
    t=t.view({-1});
    v.resize(t.size(0));
    auto t_a=t.accessor<T,1>();
    for(int i=0;i<t.size(0);i++){
        v[i]=t_a[i];
    }
}

at::Tensor vector_to_tensor(std::vector<pfloat> &v)
{
    auto options=torch::TensorOptions().dtype(torch::kFloat64);
    at::Tensor t=torch::zeros({(int)(v.size())},options);
    auto t_a=t.accessor<double,1>();
    for(int i=0;i<(int)(v.size());i++){
        t_a[i]=v[i];
    }
    return t;
}

template<class T>
void print(const std::vector<T> &vec)
{
    for(auto iter=vec.begin();iter!=vec.end();iter++){
        std::cout<<*iter<<" ";
    }
    std::cout<<std::endl;
}

PreinitData pre_data;

void init(at::Tensor m_t,at::Tensor edges_t,at::Tensor l0_t,int batch_size)
{
    std::vector<pfloat> m_v;
    tensor_to_vector<pfloat>(m_t,m_v);
    std::vector<idxint> edges_v;
    tensor_to_vector<idxint>(edges_t,edges_v);
    std::vector<pfloat> l0_v;
    tensor_to_vector<pfloat>(l0_t,l0_v);
    int n_vts=m_t.size(0);
    int n_edges=edges_t.size(0);
    opts.resize(batch_size);
    for(int i=0;i<batch_size;i++){
        opts[i].init_solver(m_v,edges_v,l0_v,n_vts,n_edges,&pre_data);
    }

    pool=std::shared_ptr<ctpl::thread_pool>(new ctpl::thread_pool(batch_size));
    init_time=std::chrono::system_clock::now();
}

void init_lap(at::Tensor Lpr_t,at::Tensor Ljc_t,at::Tensor Lir_t,double lmd_lap)
{
    pre_data.use_lap=true;
    tensor_to_vector<pfloat>(Lpr_t,pre_data.Lpr);
    tensor_to_vector<idxint>(Ljc_t,pre_data.Ljc);
    tensor_to_vector<idxint>(Lir_t,pre_data.Lir);
    pre_data.lmd_lap=lmd_lap;
}

 
void print_exit_flag_if_ncsry(int exitflag,int sample_id,bool verbose)
{
    if(verbose&&exitflag==ECOS_OPTIMAL){
        printf("sample_id:%d,ECOS_OPTIMAL\n",sample_id);
    }
    else if(exitflag==ECOS_PINF){
        printf("sample_id:%d,ECOS_PINF\n",sample_id);
    }
    else if(exitflag==ECOS_DINF){
        printf("sample_id:%d,ECOS_DINF\n",sample_id);
    }
    else if(exitflag==ECOS_INACC_OFFSET){
        printf("sample_id:%d,ECOS_INACC_OFFSET\n",sample_id);
    }
    else if(verbose&&exitflag==ECOS_MAXIT){
        printf("sample_id:%d,ECOS_MAXIT\n",sample_id);
    }
    else if(verbose&&exitflag==ECOS_NUMERICS){
        printf("sample_id:%d,ECOS_NUMERICS\n",sample_id);
    }
    else if(exitflag==ECOS_OUTCONE){
        printf("sample_id:%d,ECOS_OUTCONE\n",sample_id);
    }
    else if(exitflag==ECOS_SIGINT){
        printf("sample_id:%d,ECOS_SIGINT\n",sample_id);
    }
    else if(exitflag==ECOS_FATAL){
        printf("sample_id:%d,ECOS_FATAL\n",sample_id);
    } 
}

at::Tensor solve(at::Tensor tgt_x_t)
{
    int n_samples=tgt_x_t.size(0);
    std::vector<at::Tensor> out_xs(n_samples);
    std::vector<std::future<void> > futures(n_samples);
    bool verbose=false;
    bool use_multi_thread=true;
    auto process_sample=[&](int id,int sample_id){
        if(verbose){
            auto now=std::chrono::system_clock::now();
            printf("solve,sample_id:%d,cpu:%d,start:%f s\n",sample_id,sched_getcpu(),std::chrono::duration<double>(now-init_time).count());
        }
        at::Tensor tgt_x_ti=tgt_x_t[sample_id];

        std::vector<pfloat> tgt_x_v;
        tensor_to_vector<pfloat>(tgt_x_ti,tgt_x_v);

        EcosOpt &opt=opts[sample_id];
        std::vector<pfloat> x;
        int exitflag=opt.solve(tgt_x_v,x);
        print_exit_flag_if_ncsry(exitflag,sample_id,verbose);
        at::Tensor x_t=vector_to_tensor(x).view({1,-1,D});
        out_xs[sample_id]=x_t;
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
    return at::cat(out_xs,0);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("solve", &solve,"solve cvx opt");
  m.def("init", &init,"init cvx opt");
  m.def("init_lap",&init_lap,"init lap");
}
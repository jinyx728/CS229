//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "cvx_opt_global.h"
#include <fstream>
#include <limits>
std::shared_ptr<ctpl::thread_pool> pool=nullptr;
std::chrono::system_clock::time_point init_time;
int batch_size;
bool verbose;
bool use_multi_thread;
bool use_debug;
int forward_counter=0;
int backward_counter=0;

void tensor_to_vector(at::Tensor t,std::vector<pfloat> &v,bool resize_v)
{
    t=t.view({-1});
    if(resize_v)
        v.resize(t.size(0));
    if(t.size(0)!=v.size()){
        printf("tensor_to_vector:size does not match,t:%d,v:%d\n",(int)t.size(0),(int)v.size());
    }
    auto t_a=t.accessor<double,1>();
    for(int i=0;i<t.size(0);i++){
        v[i]=t_a[i];
    }
}
 
void tensor_to_vector(at::Tensor t,std::vector<idxint> &v,bool resize_v)
{
    t=t.view({-1});
    if(resize_v)
        v.resize(t.size(0));
    if(t.size(0)!=v.size()){
        printf("tensor_to_vector:size does not match,t:%d,v:%d\n",(int)t.size(0),(int)v.size());
    }
    auto t_a=t.accessor<idxint,1>();
    for(int i=0;i<t.size(0);i++){
        v[i]=t_a[i];
    }
}
 
at::Tensor vector_to_tensor(const std::vector<pfloat> &v)
{
    auto options=torch::TensorOptions().dtype(torch::kFloat64);
    at::Tensor t=torch::zeros({(int)(v.size())},options);
    auto t_a=t.accessor<double,1>();
    for(int i=0;i<(int)(v.size());i++){
        t_a[i]=v[i];
    }
    return t;
}

at::Tensor vector_to_tensor(const std::vector<idxint> &v)
{
    auto options=torch::TensorOptions().dtype(torch::kLong);
    at::Tensor t=torch::zeros({(int)(v.size())},options);
    auto t_a=t.accessor<idxint,1>();
    for(int i=0;i<(int)(v.size());i++){
        t_a[i]=v[i];
    }
    return t;
}

template<class T>
void print(const std::vector<T> &vec)
{
    for(auto iter=vec.begin();iter!=vec.end();iter++){
        std::cout<<(double)*iter<<" ";
    }
    std::cout<<std::endl;
} 
 
void init(int _batch_size,bool _verbose,bool _use_multi_thread,bool _use_debug)
{
    batch_size=_batch_size;
    verbose=_verbose;
    use_multi_thread=_use_multi_thread;
    use_debug=_use_debug;
    pool=std::shared_ptr<ctpl::thread_pool>(new ctpl::thread_pool(batch_size));
    init_time=std::chrono::system_clock::now();
    printf("forward:use_multi_thread:%d\n",use_multi_thread);
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
    else if(verbose&&exitflag==ECOS_INACC_OFFSET){
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

bool is_unusual_x(const std::vector<pfloat> &x)
{
    pfloat sum_sqr=0;
    for(uint i=0;i<x.size();i++){
        if(std::isnan(x[i])||std::isinf(x[i])){
            printf("x is nan or inf\n");
            return true;
        } 
        sum_sqr+=x[i]*x[i];
        if(i%D==D-1){
            if(sum_sqr>10){
                printf("unusual x\n");
                return true;
            }
            sum_sqr=0;
        }
    }
    return false;
}

bool is_unusual_dx(const std::vector<pfloat> &x)
{
    pfloat sum_sqr=0;
    for(uint i=0;i<x.size();i++){
        if(std::isnan(x[i])||std::isinf(x[i])){
            printf("x is nan or inf\n");
            return true;
        } 
        sum_sqr+=x[i]*x[i];
        if(i%D==D-1){
            if(sum_sqr>1e-1){
                printf("unusual x\n");
                return true;
            }
            sum_sqr=0;
        }
    }
}

void write_x(const std::string &file,const std::vector<pfloat> &x)
{
    std::ofstream fout(file);
    if(!fout.good()){
        printf("cannot open %s\n",file.c_str());
    }
    fout<<std::setprecision(std::numeric_limits<pfloat>::digits10+10);
    for(uint i=0;i<x.size();i++){
        fout<<x[i]<<" ";
    }
}

template void print<pfloat>(const std::vector<pfloat>&);
// template void print<double>(const std::vector<double>&);
template void print<idxint>(const std::vector<idxint>&);

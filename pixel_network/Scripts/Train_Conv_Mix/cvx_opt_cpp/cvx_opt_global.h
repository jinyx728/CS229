//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#ifndef __CVX_OPT_GLOBAL_H__
#define __CVX_OPT_GLOBAL_H__

#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "ecos_opt.h"
#include "forward_opt.h"
#include "backward_opt.h"
#include "ctpl_stl.h"

#define D 3

extern std::shared_ptr<ctpl::thread_pool> pool;
extern std::chrono::system_clock::time_point init_time;
extern int batch_size;
extern bool verbose;
extern bool use_multi_thread;
extern bool use_debug;
extern int forward_counter,backward_counter;

// cvx_opt_utils
void tensor_to_vector(at::Tensor t,std::vector<pfloat> &v,bool resize_v=true);
void tensor_to_vector(at::Tensor t,std::vector<idxint> &v,bool resize_v=true);
at::Tensor vector_to_tensor(const std::vector<pfloat> &v);
at::Tensor vector_to_tensor(const std::vector<idxint> &v);
template<class T>
void print(const std::vector<T> &vec);
void init(int _batch_size,bool _verbose,bool _use_multi_thread,bool _use_debug);
void print_exit_flag_if_ncsry(int exitflag,int sample_id,bool verbose);
bool is_unusual_x(const std::vector<pfloat> &x);
bool is_unusual_dx(const std::vector<pfloat> &dx);
void write_x(const std::string &file,const std::vector<pfloat> &x);

// cvx_opt_forward
extern std::vector<ForwardOpt> forward_opts;
extern PreinitData pre_data;
void init_forward(at::Tensor m_t,at::Tensor edges_t,at::Tensor l0_t);
void init_options(int max_it=50,pfloat feastol=1e-8,pfloat abstol=1e-8,pfloat reltol=1e-8,pfloat ftol_inacc=1e-4,pfloat atol_inacc=5e-5,pfloat rtol_inacc=5e-5);
void init_lap(at::Tensor Lpr_t,at::Tensor Ljc_t,at::Tensor Lir_t,double lmd_lap);
void init_spring(at::Tensor youngs_modulus,double lmd_k);
std::vector<at::Tensor> solve_forward(at::Tensor tgt_x_t);
std::vector<at::Tensor> solve_forward_variable_m(at::Tensor tgt_x_t,at::Tensor m_t);
void init_sols(std::vector<Solution> &sols);

// cvx_opt_backward
extern std::vector<BackwardOpt> backward_opts;
void init_backward();
at::Tensor solve_backward(at::Tensor grad_t,at::Tensor tgt_x_t,std::vector<at::Tensor> tensors);
std::vector<at::Tensor> solve_backward_variable_m(at::Tensor grad_t,at::Tensor tgt_x_t,at::Tensor m_t,std::vector<at::Tensor> tensors);

#endif
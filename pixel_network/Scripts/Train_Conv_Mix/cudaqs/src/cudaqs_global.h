//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include <torch/torch.h>
#include "tensor.h"
#include "newton_opt.h"
#include "backward_opt.h"
#include "../include/ctpl/ctpl_stl.h"

namespace cudaqs{
void init(int n_vts,int batch_size_,bool use_multi_thread_=true,bool verbose_=false);
SpringData init_spring(const at::Tensor &edges,const at::Tensor &l0,const at::Tensor &k);
AxialData init_axial(const at::Tensor &i,const at::Tensor &w,const at::Tensor &k);
SpringSystem init_system(const int n_vts,const SpringData &spring_data,const AxialData &axial_data);
NewtonOpt init_forward(const SpringSystem &system);
BackwardOpt init_backward(const SpringSystem &system,bool use_variable_stiffen_anchor);
std::vector<OptDataPtr> init_opt_data(int batch_size,int n_vts,int n_edges);
std::vector<at::Tensor> solve_forward(const NewtonOpt &forward_opt,const at::Tensor &anchor_in,const at::Tensor &stiffen_anchor_in,std::vector<OptDataPtr> &opt_datas);
std::vector<at::Tensor> solve_backward(const BackwardOpt &backward_opt,const at::Tensor &dl_in,const at::Tensor &x_in,const at::Tensor &anchor_in,const at::Tensor &stiffen_anchor_in,std::vector<OptDataPtr> &opt_datas);

extern std::shared_ptr<ctpl::thread_pool> pool;
extern int batch_size;
extern bool use_multi_thread;
extern bool use_cuda_stream;
extern bool verbose;
extern std::vector<ThreadCtx> thread_ctxs;

// extern SpringData spring_data;
// extern AxialData axial_data;
// extern SpringSystem spring_system;
// extern NewtonOpt forward_opt;

}
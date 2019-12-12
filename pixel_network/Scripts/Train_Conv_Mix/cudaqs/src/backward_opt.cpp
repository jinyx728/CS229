//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "backward_opt.h"
#include "cg_solve.h"
#include <cmath>

using namespace cudaqs;

void BackwardOpt::init(const SpringSystem *system_in, bool use_variable_stiffen_anchor_in, double cg_tol_in, int cg_max_iter_in)
{
    system=system_in;
    cg_tol=cg_tol_in;
    cg_max_iter=cg_max_iter_in;
    use_variable_stiffen_anchor=use_variable_stiffen_anchor_in; // #2
}

int BackwardOpt::solve(const Tensor<double> &dl,const Tensor<double> &x,const Tensor<double> &anchor,const Tensor<double> &stiffen_anchor,OptData &data,Tensor<double> &da,Tensor<double> &dstiffen_anchor) const
{
    SystemData &system_data=data.system_data;
    CGData &cg_data=data.cg_data;
    BackwardOptData &backward_data=data.backward_data;

    system_data.anchor.copy(anchor);
    system_data.stiffen_anchor.copy(stiffen_anchor);
    system->get_data(x,system_data);
    double sqrt_N = std::sqrt(anchor.size[0]);
    system_data.J_rms=dl.norm()/sqrt_N;

    da.set_zero();

    int cg_iters=cg_solve(system,system_data,da,dl,da,cg_data,cg_tol,cg_max_iter);
    
    int n_vts=stiffen_anchor.size[0];
    Tensor<double> &d=backward_data.resized_stiffen_anchor;
    d.copy(stiffen_anchor);
    d.size={1,n_vts};
    if(use_variable_stiffen_anchor){
        Tensor<double> &dx=backward_data.dx;
        system_data.d.negate(dx);
        dx*=da;dx.sum_col(dstiffen_anchor);
    }
    da*=d;
    d.size={n_vts};
    if(cg_iters<0){
        return MAX_CG_ITER;
    }
    else{
        return SUCCESS;
    }
}

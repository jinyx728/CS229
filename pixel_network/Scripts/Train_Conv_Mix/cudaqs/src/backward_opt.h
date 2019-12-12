//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include "spring_system.h"

namespace cudaqs{
class BackwardOpt{
public:
    static const int SUCCESS=0;
    static const int MAX_CG_ITER=1;
    BackwardOpt(){};
    void init(const SpringSystem *system,bool use_variable_stiffen_anchor_in=false,double cg_tol_in=1e-3,int cg_max_iter_in=1000);
    int solve(const Tensor<double> &dl,const Tensor<double> &x,const Tensor<double> &anchor,const Tensor<double> &stiffen_anchor,OptData &data,Tensor<double> &da,Tensor<double> &dstiffen_anchor) const;
    
    bool use_variable_stiffen_anchor;

private:
    const SpringSystem *system;
    double cg_tol;
    int cg_max_iter;
};
}
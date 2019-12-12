//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include "spring_system.h"
#include "tensor.h"

namespace cudaqs{
class NewtonOpt{
public:
    static const int SUCCESS=0;
    static const int INACCURATE_RESULT=1;
    static const int SEARCH_DIR_FAIL=2;
    static const int MAX_ITER=3;
    // Constructors
    NewtonOpt(){};
    void init(const SpringSystem *system_in, float newton_tol_in=1e-3, float cg_tol_in=1e-3, int cg_max_iter_in=1000);
    int solve(const Tensor<double> &anchor,const Tensor<double> &stiffen_anchor,OptData &data,Tensor<double> &x,int max_iter=200) const;
    double backtrack_line_search(const Tensor<double>& x0, const Tensor<double>& dx,OptData &data, double fx0, Tensor<double> &x, double t0=1, double alpha=0, double beta=0.5, int max_steps=8) const;

private:
    const SpringSystem *system;
    float newton_tol;
    float cg_tol;
    int cg_max_iter;
};
}

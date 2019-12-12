//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include "tensor.h"
#include "spring_system.h"

namespace cudaqs{

int cg_solve(const SpringSystem *system, const SystemData &A, const Tensor<double> &x0, const Tensor<double> &b, Tensor<double> &x, CGData &cg_data, float tol, int max_iter);
}

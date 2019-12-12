//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include "solver.h"
#include "la/la_eigen.h"
#include "kkt_cg_eigen.h"
#include "pwork_cg_eigen.h"

namespace ACOS{

class PolicyCGEigen{
public:
    typedef LAEigen LA;
    typedef PWorkCGEigen<LA> PWork;
    typedef KKTCGEigen<PWork> KKT;
};
}
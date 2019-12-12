//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include "pwork.h"
#include "impl/cone.h"
#include "system_cg_eigen.h"

namespace ACOS{
template<class LA>
class PWorkCGEigen:public PWork<LA>{
public:
    idxint nm,dim_z,nrhs;
    typename LA::Mat Gm,Gs,Gmt;
    typename LA::Vec Grow_invsq,Gcol_invsq;
    kkt_ldl_3x3::cone *C_impl;
    SystemCGEigen<LA> system;
};
}

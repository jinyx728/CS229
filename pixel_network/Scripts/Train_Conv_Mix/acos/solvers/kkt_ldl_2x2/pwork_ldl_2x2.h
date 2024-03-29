//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include "pwork.h"
#include "impl/kkt.h"
#include "impl/cone.h"
#include <vector>
#include <tuple>
namespace ACOS{
template<class LA>
class PWorkLDL2x2:public PWork<LA>{
public:
    kkt_ldl_3x3::kkt *kkt_impl;
    kkt_ldl_3x3::cone *C_impl;
    idxint *AtoK,*GtoK;
    kkt_ldl_3x3::spmat *A_impl,*G_impl;
    idxint nm; // number of main variables, assuming all the slack variables are at the end, nm+p=n
    // typename LA::Mat Gs,As_inv;
    // typename LA::Mat GsAs_inv;
    typename LA::Mat Gs;
    idxint nK,dim_z;
};
}

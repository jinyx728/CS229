//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include "glblopts.h"

namespace ACOS{
template<class LA>
class KKTData{
public:
    typedef typename LA::Vec Vec;
    Vec work1,rhs1,rhs2,dx1,dx2,dy1,dy2,dz1,dz2;
};
};
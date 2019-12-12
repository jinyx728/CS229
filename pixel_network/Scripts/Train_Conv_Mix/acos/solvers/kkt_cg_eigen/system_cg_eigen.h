//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include "krylov_system_base.h"
#include <memory>

namespace ACOS{
template<class LA>
class PWorkCGEigen;

template<class LA>
class SystemCGEigen:public KrylovSystemBase<LA>{
public:
    typedef typename LA::Vec Vec;
    typedef typename LA::Mat Mat;
    typedef std::shared_ptr<PWorkCGEigen<LA> > PWorkPtr;

    virtual Vec multiply(const Vec &x) const override;
    virtual pfloat inner_product(const Vec &v1,const Vec &v2) const override;
    virtual Vec precondition(const Vec &x) const override;
    virtual pfloat convergence_norm(const Vec &r) const override;
    Vec W2_inv(const Vec &x) const;
    Vec W2(const Vec &x) const;
    Vec Jacobi(const Vec &x,const Vec &D) const;
    PWorkPtr w; 
    idxint isinit;

    SystemCGEigen():w(nullptr),isinit(1){}
};
}
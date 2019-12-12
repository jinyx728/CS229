//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include "glblopts.h"
#include <memory>

namespace ACOS{
template<class PWork>
class KKTBase{
public:
    typedef typename PWork::LA_type LA;
    typedef typename LA::Vec Vec;
    typedef typename LA::Mat Mat;
    typedef typename PWork::ConePtr ConePtr;
    typedef std::shared_ptr<PWork> PWorkPtr;
    enum KKTStat{
        KKT_PROBLEM,KKT_OK
    };
    enum ConeStat{
        INSIDE_CONE,OUTSIDE_CONE
    };
    
    static void init(PWorkPtr w);
    static void init_rhs1(PWorkPtr w,Vec &rhs);
    static void init_rhs2(PWorkPtr w,Vec &rhs);
    static void update_rhs1(PWorkPtr w,Vec &rhs);
    static KKTStat factor(PWorkPtr w);
    static void solve(PWorkPtr w,Vec &rhs,Vec &dx,Vec &dy,Vec &dz);
    static void update(PWorkPtr w,ConePtr C);
    static void RHS_affine(PWorkPtr w,Vec &rhs);
    static void RHS_combined(PWorkPtr w,Vec &rhs);
    static pfloat lineSearch(PWorkPtr w);
    static void cleanup(PWorkPtr w);
    static void scale(Vec &z,ConePtr C,Vec &lambda);
    static void bring2cone(ConePtr C,Vec &r,Vec &s);
    static ConeStat updateScalings(ConePtr C,Vec &s,Vec &z,Vec &lambda);
};
}
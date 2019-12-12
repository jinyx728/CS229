//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include "glblopts.h"
#include "kkt_base.h"

namespace ACOS{
template<class PWork>
class KKTCGEigen:public KKTBase<PWork>{
public:
    typedef KKTBase<PWork> Base;
    using typename Base::LA;using typename Base::Vec;using typename Base::Mat;using typename Base::ConePtr;using typename Base::PWorkPtr;using typename Base::KKTStat;using typename Base::ConeStat;
    static void init(PWorkPtr w);
    static void init_rhs1(PWorkPtr w,Vec &rhs);
    static void init_rhs2(PWorkPtr w,Vec &rhs);
    static void update_rhs1(PWorkPtr w,Vec &rhs);
    static KKTStat factor(PWorkPtr w);
    static void solve(PWorkPtr w,Vec &rhs,Vec &dx,Vec &dy,Vec &dz,idxint isinit);
    static void update(PWorkPtr w,ConePtr C);
    static void RHS_affine(PWorkPtr w,Vec &rhs);
    static void RHS_combined(PWorkPtr w,Vec &rhs);
    static pfloat lineSearch(PWorkPtr w, Vec &lambda, Vec &ds, Vec &dz, pfloat tau, pfloat dtau, pfloat kap, pfloat dkap);
    static void cleanup(PWorkPtr w);
    static void scale(PWorkPtr w, Vec &z,Vec &lambda);
    static void bring2cone(PWorkPtr w,Vec &r,Vec &s);
    static ConeStat updateScalings(PWorkPtr w,Vec &s,Vec &z,Vec &lambda);
    static void set_equil(PWorkPtr w);
    static void backscale(PWorkPtr w);
    static void get_row_invsq(const Mat &m,Vec &v);
    static void get_col_invsq(const Mat &m,Vec &v);
};
}
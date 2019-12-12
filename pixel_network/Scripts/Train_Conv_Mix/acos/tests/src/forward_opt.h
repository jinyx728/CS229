//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include "glblopts.h"
#include <vector>

#define D 3
struct Solution{
    std::vector<pfloat> x,y,z,s;
    idxint success;
};

class ForwardOpt{
public:
    ForwardOpt();
    ~ForwardOpt();
    void init_solver(const std::vector<pfloat> &m,const std::vector<idxint> &edges,const std::vector<pfloat> &l0,const int n_vts,const int n_edges);
    template<class Solver>
    int solve(const std::vector<pfloat> &tgt_x,Solution &sol,bool verbose=false);
    
    void create_G(const std::vector<idxint> &edges,const std::vector<pfloat> &w);
    void create_h();
    void create_A();
    void create_b(const std::vector<pfloat> &l0);

    void convert_CCS(const std::vector<std::vector<pfloat> > &pr,const std::vector<std::vector<idxint> > &ir,std::vector<pfloat> &Xpr,std::vector<idxint> &Xjc,std::vector<idxint> &Xir) const;

    int n,m,p,l,ncones,nex;
    std::vector<idxint> q;
    std::vector<pfloat> c,h,b,x0;
    std::vector<pfloat> Gpr;std::vector<idxint> Gjc,Gir;
    std::vector<pfloat> Apr;std::vector<idxint> Ajc,Air;

    int n_vts,n_edges;
    std::vector<pfloat> sqrt_w;
    int x_offset,t_offset,s_offset,l_offset;

    // for debug
    std::vector<idxint> edges;

};


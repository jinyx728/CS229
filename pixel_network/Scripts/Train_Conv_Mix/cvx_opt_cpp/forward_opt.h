//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#ifndef __FORWARD_SOLVE_H__
#define __FORWARD_SOLVE_H__
#include <Eigen/SparseCore>
#include "ecos_opt.h"
#include <tuple>

struct PreinitData{
    bool use_lap;pfloat lmd_lap;
    std::vector<pfloat> Lpr;std::vector<idxint> Ljc,Lir;

    bool use_spring;
    std::vector<pfloat> youngs_modulus;pfloat lmd_k;

    PreinitData():use_lap(false),lmd_lap(1.0),use_spring(false),lmd_k(1.0){};
};

struct Solution{
    std::vector<pfloat> x,y,z,s;
    idxint success;
};

typedef Eigen::SparseMatrix<double,Eigen::ColMajor> SpaMat;

class ForwardOpt:public EcosOpt{
public:
    typedef EcosOpt Base;
    using Base::n;using Base::m;using Base::p;using Base::l;using Base::ncones;using Base::nex;
    using Base::q;using Base::c;using Base::h;using Base::b;using Base::Gpr;using Base::Gjc;using Base::Gir;using Base::Apr;using Base::Ajc;using Base::Air;
    using Base::mwork;
    ForwardOpt();
    ~ForwardOpt();
    void init_solver(const std::vector<pfloat> &m,const std::vector<idxint> &edges,const std::vector<pfloat> &l0,const int n_vts,const int n_edges,PreinitData *pre_data);
    int solve(const std::vector<pfloat> &tgt_x,Solution &sol,bool verbose=false);
    int solve(const std::vector<pfloat> &tgt_x,const std::vector<pfloat> &w,Solution &sol,bool verbose=false);
    void check(const std::vector<pfloat> &x,const std::vector<pfloat> &tgt_x,const std::vector<pfloat> &lmd);
    void check_sol(const Solution &sol) const;
    
    void create_G(const std::vector<idxint> &edges,const std::vector<pfloat> &w);
    void create_h();
    void create_A();
    void create_b(const std::vector<pfloat> &l0);

    int n_vts,n_edges;
    std::vector<pfloat> sqrt_w;
    int x_offset,t_offset,s_offset,l_offset;

    PreinitData *pre_data;
    pfloat relax_factor;

    // lap
    idxint create_G_lap(idxint row_id,std::vector<std::vector<pfloat> > &pr,std::vector<std::vector<idxint> > &ir);
    void update_h_lap(const std::vector<pfloat> &x);
    bool use_lap;
    SpaMat L;

    // springs
    typedef std::tuple<idxint,idxint> range;

    idxint create_G_spring(idxint row_id,std::vector<std::vector<pfloat> > &pr,std::vector<std::vector<idxint> > &ir);
    idxint create_A_spring(idxint row_id,const std::vector<idxint> &edges,std::vector<std::vector<pfloat> > &pr,std::vector<std::vector<idxint> > &ir);
    void update_G_h_spring(const std::vector<pfloat> &x,const std::vector<std::vector<idxint> > &spring_id_to_G);
    void get_id_to_G(const range &r,std::vector<std::vector<idxint> > &id_to_G);
    void get_stiffness(const std::vector<pfloat> &youngs_modulus);
    bool use_spring;
    range spring_range;idxint dx_offset;
    range G_spring_range,A_spring_range;
    std::vector<std::vector<idxint> > spring_id_to_G;
    std::vector<pfloat> sqrt_stiffness;

    // variable m
    void update_w_G(const std::vector<pfloat> &w);
    idxint find_ccs_id(const int row,const int col,const std::vector<idxint> &ir,const std::vector<idxint> &jc);
    void compute_m_id_to_G();
    std::vector<idxint> m_id_to_G;

    // for debug
    std::vector<idxint> edges;
    std::vector<pfloat> l0;
};

Eigen::VectorXd Eigen_From_Raw(const pfloat *data,const int n);
void create_ccs_mat(const int nrows,const int ncols,const std::vector<pfloat> &Lpr,const std::vector<idxint> &Ljc,const std::vector<idxint> &Lir,SpaMat &L);
void create_L(const std::vector<pfloat> &Lpr,const std::vector<idxint> &Ljc,const std::vector<idxint> &Lir,SpaMat &L);

#endif
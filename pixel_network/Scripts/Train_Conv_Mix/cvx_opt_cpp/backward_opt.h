//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#ifndef __BACKWARD_OPT_H__
#define __BACKWARD_OPT_H__
#include "forward_opt.h"
#include <Eigen/SparseCore>

class BackwardOpt:public EcosOpt{
public:
    typedef Eigen::SparseMatrix<double,Eigen::ColMajor> SpaMat;

    BackwardOpt();
    ~BackwardOpt();
    void setup();
    void update();

    void init_solver(ForwardOpt *_forward_opt);
    idxint solve(const std::vector<pfloat> &tgt_x,const Solution &sol,const std::vector<pfloat> &in_grad,std::vector<pfloat> &out_grad);
    idxint solve(const std::vector<pfloat> &tgt_x,const std::vector<pfloat> &w,const Solution &sol,const std::vector<pfloat> &in_grad,std::vector<pfloat> &out_grad,std::vector<pfloat> &out_m_grad);

    void copy_and_prescale(pwork *mwork,const std::vector<pfloat> &x,const std::vector<pfloat> &y,const std::vector<pfloat> &z,const std::vector<pfloat> &s);
    void scale_and_get_RHSx(pwork *mwork,const std::vector<pfloat> &in_grad,std::vector<pfloat> &RHSx);
    void scale_and_get_dx(pwork *mwork,const std::vector<pfloat> &dz,std::vector<pfloat> &dx);
    void backscale(const pwork *mwork,std::vector<pfloat> &dx,std::vector<pfloat> &dy,std::vector<pfloat> &dz);
    void dh_dxj(const std::vector<pfloat> &dh,std::vector<pfloat> &dxj);
    void dG_dxj(const std::vector<pfloat> &x,const std::vector<pfloat> &z,const std::vector<pfloat> &dx,const std::vector<pfloat> &dz,const std::vector<pfloat> &xj,std::vector<pfloat> &dxj);
    void dh_dmj(const std::vector<pfloat> &dh,const std::vector<pfloat> &tgt_x,std::vector<pfloat> &dmj);
    void dG_dmj(const std::vector<pfloat> &x,const std::vector<pfloat> &z,const std::vector<pfloat> &dx,const std::vector<pfloat> &dz,std::vector<pfloat> &dm);

    int n_vts,n_edges;

    bool use_spring;

    bool use_lap;
    SpaMat Lt;
    std::vector<pfloat> sqrt_w;
    ForwardOpt *forward_opt;

    std::vector<pfloat> c_work,h_work,b_work;
    std::vector<pfloat> Gpr_work;std::vector<idxint> Gjc_work,Gir_work;
    std::vector<pfloat> Apr_work;std::vector<idxint> Ajc_work,Air_work;

};

#endif
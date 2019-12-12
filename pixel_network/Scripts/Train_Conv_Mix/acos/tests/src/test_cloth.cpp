//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "solver.h"
#include "solvers/kkt_ldl_3x3/policy_ldl_3x3.h"
#include "solvers/kkt_ldl_2x2/policy_ldl_2x2.h"
#include "solvers/kkt_cg_eigen/policy_cg_eigen.h"
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <sstream>

#include "forward_opt.h"
#include "forward_opt.cpp"
#include "io_utils.h"

// typedef ACOS::Solver<ACOS::PolicyLDL3x3> Solver;
// typedef ACOS::Solver<ACOS::PolicyLDL2x2> Solver;
typedef ACOS::Solver<ACOS::PolicyCGEigen> Solver;
template int ForwardOpt::solve<Solver>(const std::vector<pfloat> &tgt_x,Solution &sol,bool verbose=false);

int main()
{
    std::string edge_path="../data/p13/edges.txt";
    std::vector<idxint> edges;read_txt(edge_path,edges);
    std::string m_path="../data/p13/m.txt";
    std::vector<pfloat> m;read_txt(m_path,m);
    std::string l0_path="../data/p13/l0.txt";
    std::vector<pfloat> l0;read_txt(l0_path,l0);
    std::cout<<"edges:"<<edges.size()<<",m:"<<m.size()<<",l0:"<<l0.size()<<std::endl;
    int n_vts=m.size(),n_edges=edges.size()/2;
    ForwardOpt opt;opt.init_solver(m,edges,l0,n_vts,n_edges);
    std::string tgt_path="../data/p13/pd_00000106.txt";
    std::vector<pfloat> tgt_x;read_txt(tgt_path,tgt_x);
    std::string fcs_path="../data/p13/fcs.txt";
    std::vector<idxint> fcs;read_txt(fcs_path,fcs);
    std::cout<<"tgt_x:"<<tgt_x.size()<<",fcs:"<<fcs.size()<<std::endl;

    Solution sol;
    sol.x.resize(opt.n);sol.y.resize(opt.p);sol.z.resize(opt.m);sol.s.resize(opt.m);
    opt.solve<Solver>(tgt_x,sol,true);

    std::string out_path="p13_cr_00000106.obj";
    std::vector<pfloat> cr_x(n_vts*D);
    std::copy(sol.x.begin(),sol.x.begin()+n_vts*D,cr_x.begin());
    write_obj(out_path,cr_x,fcs);

    return 0;
}
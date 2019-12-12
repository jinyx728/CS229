//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "forward_opt.h"
#include "io_utils.h"

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
    std::string tgt_path="../data/p13/pd_00000106.txt";
    std::vector<pfloat> tgt_x;read_txt(tgt_path,tgt_x);
    std::string fcs_path="../data/p13/fcs.txt";
    std::vector<idxint> fcs;read_txt(fcs_path,fcs);
    std::cout<<"tgt_x:"<<tgt_x.size()<<",fcs:"<<fcs.size()<<std::endl;

    PreinitData pre_data;
    bool use_spring=true;
    if(use_spring){
        std::string k_path="../data/p13/k.txt";
        read_txt(k_path,pre_data.youngs_modulus);
        std::cout<<",k:"<<pre_data.youngs_modulus.size()<<std::endl;
        pre_data.use_spring=true;
    }

    ForwardOpt opt;opt.init_solver(m,edges,l0,n_vts,n_edges,&pre_data);

    Solution sol;
    sol.x.resize(opt.n);sol.y.resize(opt.p);sol.z.resize(opt.m);sol.s.resize(opt.m);
    // opt.solve(tgt_x,sol,true);
    opt.solve(tgt_x,m,sol,true);

    std::string out_path="out/p13/cr_00000106.obj";
    std::vector<pfloat> cr_x(n_vts*D);
    std::copy(sol.x.begin(),sol.x.begin()+n_vts*D,cr_x.begin());
    write_obj(out_path,cr_x,fcs);

    out_path="out/p13/pd_00000106.obj";
    write_obj(out_path,tgt_x,fcs);

    write_txt("out/p13/x_00000106.txt",sol.x);
    write_txt("out/p13/y_00000106.txt",sol.y);
    write_txt("out/p13/z_00000106.txt",sol.z);
    write_txt("out/p13/s_00000106.txt",sol.s);
    write_txt("out/p13/tgt_x_00000106.txt",tgt_x);

    return 0;
}
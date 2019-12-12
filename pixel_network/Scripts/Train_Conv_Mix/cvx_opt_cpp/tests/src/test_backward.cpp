//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "backward_opt.h"
#include "io_utils.h"

int main()
{
    std::string data_dir="../data/whole/";
    std::string out_dir="out/whole";
    std::string id_str="00015037";
    std::string edge_path=data_dir+"/edges.txt";
    std::vector<idxint> edges;read_txt(edge_path,edges);
    std::string m_path=data_dir+"/m.txt";
    std::vector<pfloat> m;read_txt(m_path,m);
    std::string l0_path=data_dir+"/l0.txt";
    std::vector<pfloat> l0;read_txt(l0_path,l0);
    std::cout<<"edges:"<<edges.size()<<",m:"<<m.size()<<",l0:"<<l0.size()<<std::endl;
    int n_vts=m.size(),n_edges=edges.size()/2;
    std::string tgt_path=data_dir+"/pd_"+id_str+".txt";
    std::vector<pfloat> tgt_x;read_txt(tgt_path,tgt_x);
    std::string fcs_path=data_dir+"/fcs.txt";
    std::vector<idxint> fcs;read_txt(fcs_path,fcs);
    std::cout<<"tgt_x:"<<tgt_x.size()<<",fcs:"<<fcs.size()<<std::endl;

    PreinitData pre_data;
    bool use_spring=false;
    if(use_spring){
        std::string k_path=data_dir+"/k.txt";
        read_txt(k_path,pre_data.youngs_modulus);
        std::cout<<",k:"<<pre_data.youngs_modulus.size()<<std::endl;
        pre_data.use_spring=true;
    }

    ForwardOpt forward_opt;forward_opt.init_solver(m,edges,l0,n_vts,n_edges,&pre_data);

    Solution sol;
    read_txt(out_dir+"/x_"+id_str+".txt",sol.x);
    read_txt(out_dir+"/y_"+id_str+".txt",sol.y);
    read_txt(out_dir+"/z_"+id_str+".txt",sol.z);
    read_txt(out_dir+"/s_"+id_str+".txt",sol.s);
    std::vector<pfloat> cr_x(n_vts*D);std::copy(sol.x.begin(),sol.x.begin()+n_vts*D,cr_x.begin());
    std::vector<pfloat> gt_x;
    read_txt(data_dir+"/gt_"+id_str+".txt",gt_x);
    std::vector<pfloat> in_grad(n_vts*D);
    for(uint i=0;i<in_grad.size();i++)
        in_grad[i]=cr_x[i]-gt_x[i];
    sol.success=true;

    BackwardOpt backward_opt;backward_opt.init_solver(&forward_opt);
    // std::vector<pfloat> out_grad;
    // backward_opt.solve(tgt_x,sol,in_grad,out_grad);

    std::vector<pfloat> out_grad,out_m_grad;
    backward_opt.solve(tgt_x,m,sol,in_grad,out_grad,out_m_grad);

    write_txt(out_dir+"/grad_"+id_str+".txt",out_grad);
    std::cout<<"write to "+out_dir+"/grad_"+id_str+".txt"<<std::endl;
    write_txt(out_dir+"/grad_m_"+id_str+".txt",out_m_grad);
    return 0;
}
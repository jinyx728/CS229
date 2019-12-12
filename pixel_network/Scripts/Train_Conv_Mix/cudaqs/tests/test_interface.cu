//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <vector>
#include <stdio.h>
#include <chrono>

#include "io_utils.h"
#include "newton_opt.h"
#include "cudaqs_global.h"
#include "cudaqs_utils.h"

using namespace cudaqs;

int main()
{
    auto options_int=torch::TensorOptions().device(torch::kCUDA,0).dtype(torch::kInt32);
    auto options_double=torch::TensorOptions().device(torch::kCUDA,0).dtype(torch::kFloat64);
    int batch_size=16;

    std::string data_dir="../data/";
    SpringData spring_data=load_spring_data(data_dir);
    AxialData axial_data=load_axial_data(data_dir);

    Tensor<double> anchor_t(0),stiffen_anchor_t(0);
    load_anchor_data(data_dir,anchor_t,stiffen_anchor_t);
    at::Tensor anchor=torch::empty({1},options_double);
    to_torch_tensor<double>(anchor_t,anchor);
    anchor=anchor.permute({1,0});
    anchor=anchor.unsqueeze(0).repeat({batch_size,1,1}).contiguous();
    at::Tensor stiffen_anchor=torch::empty({1},options_double);
    to_torch_tensor<double>(stiffen_anchor_t,stiffen_anchor);
    stiffen_anchor=stiffen_anchor.unsqueeze(0).repeat({batch_size,1}).contiguous();

    at::Tensor edges=torch::empty({1},options_int);
    to_torch_tensor<int>(spring_data.edges,edges);edges=edges.permute({1,0}).contiguous();
    at::Tensor l0=torch::empty({1},options_double);
    to_torch_tensor<double>(spring_data.l0,l0);l0=l0.contiguous();
    at::Tensor k=torch::empty({1},options_double);
    to_torch_tensor<double>(spring_data.k,k);k.contiguous();
    spring_data=init_spring(edges,l0,k);

    at::Tensor axial_i=torch::empty({1},options_int);to_torch_tensor<int>(axial_data.i,axial_i);axial_i=axial_i.permute({1,0}).contiguous();
    at::Tensor axial_w=torch::empty({1},options_double);to_torch_tensor<double>(axial_data.w,axial_w);axial_w=axial_w.permute({1,0}).contiguous();
    at::Tensor axial_k=torch::empty({1},options_double);to_torch_tensor<double>(axial_data.k,axial_k);axial_k=axial_k.contiguous();
    axial_data=init_axial(axial_i,axial_w,axial_k);

    int n_vts=stiffen_anchor_t.size[0];
    bool use_multi_thread=true;
    bool verbose=true;
    init(n_vts,batch_size,use_multi_thread,verbose);

    SpringSystem system=init_system(n_vts,spring_data,axial_data);
    NewtonOpt forward_opt=init_forward(system);
    std::vector<OptDataPtr> opt_datas=init_opt_data(batch_size,n_vts,spring_data.n_edges);
    at::Tensor xt=solve_forward(forward_opt,anchor,stiffen_anchor,opt_datas)[0];
    std::cout<<"norm:"<<torch::norm(xt)<<std::endl;

    std::string fcs_path=data_dir+"/fcs.txt";
    std::vector<int> fcs;read_txt(fcs_path,fcs);
    Tensor<double> x(0);from_torch_tensor<double>(xt[0],x);
    std::vector<double> cr_v(n_vts*3);
    thrust::copy(x.v.begin(),x.v.end(),cr_v.begin());
    std::string obj_path="out/whole/cr_00000106.obj";
    std::cout<<"write to "<<obj_path<<std::endl;
    write_obj(obj_path,cr_v,fcs);

}
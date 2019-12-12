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
#include "backward_opt.h"
using namespace cudaqs;

int main()
{
    ThreadCtx ctx;
    cudaStreamCreateWithFlags(&ctx.stream,cudaStreamNonBlocking);
    // cudaStreamCreate(&ctx.stream);
    cublasCreate(&ctx.handle);
    ThreadCtx *ctx_ptr=&ctx;

    std::string data_dir="../data/";
    SpringData spring_data=load_spring_data(data_dir);
    AxialData axial_data=load_axial_data(data_dir);
    Tensor<double> anchor(ctx_ptr),stiffen_anchor(ctx_ptr);
    load_anchor_data(data_dir,anchor,stiffen_anchor);
    int n_vts=stiffen_anchor.size[0];

    std::string cr_path=data_dir+"/cr_00000106.txt";
    Tensor<double> cr(ctx_ptr);read_txt_and_transpose(cr_path,cr,{n_vts,3});
    std::string gt_path=data_dir+"/gt_00000106.txt";
    Tensor<double> gt(ctx_ptr);read_txt_and_transpose(gt_path,gt,{n_vts,3});
    //Tensor<double> dl=cr-gt;
    Tensor<double> dl(gt.size,ctx_ptr);
    cr.subtract(gt, dl);

    std::string fcs_path=data_dir+"/fcs.txt";
    std::vector<int> fcs;read_txt(fcs_path,fcs);

    SpringSystem system;system.init(n_vts,spring_data,axial_data);
    bool use_variable_stiffen_anchor=true;
    BackwardOpt opt;opt.init(&system,use_variable_stiffen_anchor);
    Tensor<double> da(anchor.size,ctx_ptr);
    Tensor<double> dstiffen_anchor({n_vts},ctx_ptr);
    OptData data(ctx_ptr);data.resize(anchor.size,spring_data.n_edges);

    auto start=std::chrono::system_clock::now();
    opt.solve(dl,cr,anchor,stiffen_anchor,data,da,dstiffen_anchor);
    auto end=std::chrono::system_clock::now();
    printf("duration:%fs\n",std::chrono::duration<double>(end-start).count());

    printf("da:%f\n",da.norm());

    std::vector<double> da_vt(n_vts*3);
    thrust::copy(da.v.begin(),da.v.end(),da_vt.begin());
    std::vector<int> size={3,n_vts};
    std::vector<double> da_v;transpose<double>(da_vt,da_v,size);
    std::string txt_path="out/whole/grad_00000106.txt";
    printf("write to:%s\n",txt_path.c_str());
    write_txt(txt_path,da_v);

    if(use_variable_stiffen_anchor){
        printf("dstiffen_anchor:%f\n",dstiffen_anchor.norm());
        std::vector<double> dstiffen_anchor_v(n_vts);
        thrust::copy(dstiffen_anchor.v.begin(),dstiffen_anchor.v.end(),dstiffen_anchor_v.begin());
        std::string txt_path="out/whole/grad_m_00000106.txt";
        printf("write to:%s\n",txt_path.c_str());
        write_txt(txt_path,dstiffen_anchor_v);
    }
}

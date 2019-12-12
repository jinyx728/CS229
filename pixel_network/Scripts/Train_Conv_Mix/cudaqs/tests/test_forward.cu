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

using namespace cudaqs;

void get_device_info()
{
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      printf("  Memory Clock Rate (KHz): %d\n",
             prop.memoryClockRate);
      printf("  Memory Bus Width (bits): %d\n",
             prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
             2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}

int main()
{
    //get_device_info();

    std::string data_dir="../data/";
    SpringData spring_data=load_spring_data(data_dir);
    AxialData axial_data=load_axial_data(data_dir);

    ThreadCtx ctx;
    cudaStreamCreateWithFlags(&ctx.stream,cudaStreamNonBlocking);
    cublasCreate(&ctx.handle);
    ThreadCtx *ctx_ptr=&ctx;
    Tensor<double> anchor(ctx_ptr),stiffen_anchor(ctx_ptr);
    load_anchor_data(data_dir,anchor,stiffen_anchor);

    std::string fcs_path=data_dir+"/fcs.txt";
    std::vector<int> fcs;read_txt(fcs_path,fcs);

    int n_vts=stiffen_anchor.size[0];
    SpringSystem system;system.init(n_vts,spring_data,axial_data);
    NewtonOpt opt;opt.init(&system);
    Tensor<double> x(anchor.size,ctx_ptr);
    OptData data(ctx_ptr);data.resize(x.size,spring_data.n_edges);

    auto start=std::chrono::system_clock::now();
    opt.solve(anchor,stiffen_anchor,data,x);
    auto end=std::chrono::system_clock::now();
    printf("duration:%fs\n",std::chrono::duration<double>(end-start).count());

    printf("x:%f\n",x.norm());

    std::vector<double> cr_vt(n_vts*3);
    thrust::copy(x.v.begin(),x.v.end(),cr_vt.begin());
    std::vector<int> size={3,n_vts};
    std::vector<double> cr_v;transpose<double>(cr_vt,cr_v,size);
    std::string txt_path=data_dir+"/cr_00000106.txt";
    std::cout<<"write to "<<txt_path<<std::endl;
    write_txt(txt_path,cr_v);
    std::string obj_path="out/whole/cr_00000106.obj";
    std::cout<<"write to "<<obj_path<<std::endl;
    write_obj(obj_path,cr_v,fcs);
}

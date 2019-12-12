//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include "tensor.h"
namespace cudaqs{
class SpringData{
public:
    Tensor<int> edges;
    Tensor<double> l0;
    Tensor<double> k;
    int n_edges;
    SpringData():edges(nullptr),l0(nullptr),k(nullptr),n_edges(0){};
};

class AxialData{
public:
    Tensor<int> i;
    Tensor<double> w;
    Tensor<double> k;
    int n_edges;
    AxialData():i(nullptr),w(nullptr),k(nullptr),n_edges(0){};
};

class SystemData{
public:
    const ThreadCtx *ctx_ptr;
    Tensor<double> x;
    Tensor<double> d,l,lhat;
    Tensor<double> l0_over_l;

    Tensor<double> anchor;
    Tensor<double> stiffen_anchor;

    double J_rms;

    SystemData(const ThreadCtx *ctx_ptr_input);
    void resize(const std::vector<int> &size,int n_edges);
};

class CGData{
public:
    const ThreadCtx *ctx_ptr;
    Tensor<double> r,mr,q,s,buffer,Ax,n;

    CGData(const ThreadCtx *ctx_ptr_input);
    void resize(const std::vector<int> &size);
};

class NewtonOptData{
public:
    const ThreadCtx *ctx_ptr;
    Tensor<double> dx,J,neg_J,x0,min_x_buffer,Hdx,Ax;
    Tensor<double> anchor,stiffen_anchor,x;
    NewtonOptData(const ThreadCtx *ctx_ptr_input);
    void resize(const std::vector<int> &size);
};

class BackwardOptData{
public:
    const ThreadCtx *ctx_ptr;
    Tensor<double> resized_stiffen_anchor;
    Tensor<double> dl,da,dx,dstiffen_anchor;
    BackwardOptData(const ThreadCtx *ctx_ptr_input);
    void resize(const std::vector<int> &size);
};

class OptData{
public:
    const ThreadCtx *ctx_ptr;
    SystemData system_data;
    CGData cg_data;
    NewtonOptData forward_data;
    BackwardOptData backward_data;

    OptData(const ThreadCtx *ctx_ptr_input);
    void resize(const std::vector<int> x_size,const int n_edges);
};

typedef std::shared_ptr<OptData> OptDataPtr; 
}
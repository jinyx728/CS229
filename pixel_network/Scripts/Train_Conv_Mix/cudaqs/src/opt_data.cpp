//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "opt_data.h"
using namespace cudaqs;

SystemData::SystemData(const ThreadCtx *ctx_ptr_input):ctx_ptr(ctx_ptr_input),x(ctx_ptr_input),d(ctx_ptr_input),l(ctx_ptr_input),lhat(ctx_ptr_input),l0_over_l(ctx_ptr_input),anchor(ctx_ptr_input),stiffen_anchor(ctx_ptr_input)
{}

void SystemData::resize(const std::vector<int> &x_size,int n_edges)
{
    x.resize(x_size);int D=x_size[0],n_vts=x_size[1];
    d.resize({D,n_edges});
    l.resize({n_edges});
    lhat.resize({D,n_edges});
    l0_over_l.resize({n_edges});
    anchor.resize(x_size);
    stiffen_anchor.resize({n_vts});
}

CGData::CGData(const ThreadCtx *ctx_ptr_input):ctx_ptr(ctx_ptr_input),r(ctx_ptr_input),mr(ctx_ptr_input),q(ctx_ptr_input),s(ctx_ptr_input),buffer(ctx_ptr_input),Ax(ctx_ptr_input),n(ctx_ptr_input)
{}

void CGData::resize(const std::vector<int> &size)
{
    int n_vts=size[1];
    r.resize(size);mr.resize(size);q.resize(size);s.resize(size);buffer.resize(size);Ax.resize(size);n.resize({n_vts});
}

NewtonOptData::NewtonOptData(const ThreadCtx *ctx_ptr_input):ctx_ptr(ctx_ptr_input),dx(ctx_ptr_input),J(ctx_ptr_input),neg_J(ctx_ptr_input),x0(ctx_ptr_input),min_x_buffer(ctx_ptr_input),Hdx(ctx_ptr_input),Ax(ctx_ptr_input),anchor(ctx_ptr_input),stiffen_anchor(ctx_ptr_input),x(ctx_ptr_input)
{}

void NewtonOptData::resize(const std::vector<int> &x_size)
{
    dx.resize(x_size);J.resize(x_size);neg_J.resize(x_size);x0.resize(x_size);min_x_buffer.resize(x_size);Hdx.resize(x_size);Ax.resize(x_size);
    int n_vts=x_size[1];
    anchor.resize(x_size);stiffen_anchor.resize({n_vts});x.resize(x_size);
}

BackwardOptData::BackwardOptData(const ThreadCtx *ctx_ptr_input):resized_stiffen_anchor(ctx_ptr_input),dl(ctx_ptr_input),da(ctx_ptr_input),dx(ctx_ptr_input),dstiffen_anchor(ctx_ptr_input)
{}

void BackwardOptData::resize(const std::vector<int> &x_size)
{
    int n_vts=x_size[1];
    resized_stiffen_anchor.resize({n_vts});
    dl.resize(x_size);da.resize(x_size);dx.resize(x_size);dstiffen_anchor.resize({n_vts});
}

OptData::OptData(const ThreadCtx *ctx_ptr_input):ctx_ptr(ctx_ptr_input),system_data(ctx_ptr_input),cg_data(ctx_ptr_input),forward_data(ctx_ptr_input),backward_data(ctx_ptr_input)
{}

void OptData::resize(const std::vector<int> x_size,const int n_edges)
{
    system_data.resize(x_size,n_edges);
    cg_data.resize(x_size);
    forward_data.resize(x_size);
    backward_data.resize(x_size);
}
//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include "tensor.h"
#include "opt_data.h"
namespace cudaqs{
class SpringSystem{
public:
    void init(int n_vts_input,const SpringData &spring_data_input,const AxialData &axial_data_input);

    void get_data(const Tensor<double> &x,SystemData &opt_data) const;
    void get_J(const SystemData &opt_data,Tensor<double> &J) const;
    void get_Hu(const SystemData &opt_data,const Tensor<double> &u,Tensor<double> &Hu) const;
    void mul(const SystemData &opt_data,const Tensor<double> &u,Tensor<double> &Hu) const;
    double inner(const Tensor<double> &x,const Tensor<double> &y) const;
    double convergence_norm(const SystemData &data,const Tensor<double> &r, Tensor<double> &n) const;
    void precondition(const SystemData &data,const Tensor<double> &in_x,Tensor<double> &out_x) const;

    int n_vts;
    SpringData spring_data;
    AxialData axial_data;
};

}

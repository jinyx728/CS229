//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include <string>
#include <vector>
#include "spring_system.h"

namespace cudaqs{
template<class T>
void read_txt(std::string path,std::vector<T> &v);
template<class T>
void write_txt(std::string path,const std::vector<T> &v);
void read_txt_and_transpose(std::string path,Tensor<double> &t,std::vector<int> size);

void read_obj(const std::string obj_path,std::vector<double> &vts,std::vector<int> &fcs);
void write_obj(const std::string obj_path,const std::vector<double> &vts,const std::vector<int> &fcs);

template<class T>
void transpose(const std::vector<T> &in_v,std::vector<T> &out_v,std::vector<int> &size);

SpringData load_spring_data(std::string data_dir);
AxialData load_axial_data(std::string data_dir);
void load_anchor_data(std::string data_dir,Tensor<double> &anchor,Tensor<double> &stiffen_anchor);
}


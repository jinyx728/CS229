//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include <string>
#include <vector>
#include "glblopts.h"

template<class T>
void read_txt(std::string path,std::vector<T> &v);
template<class T>
void write_txt(std::string path,const std::vector<T> &v);

void read_obj(const std::string obj_path,std::vector<pfloat> &vts,std::vector<idxint> &fcs);
void write_obj(const std::string obj_path,const std::vector<pfloat> &vts,const std::vector<idxint> &fcs);

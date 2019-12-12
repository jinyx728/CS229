//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once

#include <torch/torch.h>
#include "tensor.h"

namespace cudaqs{
template<class T>
void from_torch_tensor(const at::Tensor &in,Tensor<T> &out,bool new_tensor=true);

template<class T>
void to_torch_tensor(const Tensor<T> &in,at::Tensor &out,bool new_tensor=true);
}
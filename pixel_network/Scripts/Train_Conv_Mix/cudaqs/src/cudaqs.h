//#####################################################################
// Copyright 2019. Dan Johnson. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#ifndef CUDAQS_H
#define CUDAQS_H

#include <torch/extension.h>

std::vector<double> row_sum(torch::Tensor input);
std::vector<double> row_sum_cuda(torch::Tensor input);
std::vector<float> row_sum_ctpl(torch::Tensor input, int num_ctpl_threads);
std::vector<float> row_sum_cuda_ctpl(torch::Tensor input, int num_ctpl_threads);

#endif

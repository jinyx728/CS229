//#####################################################################
// Copyright 2019. Dan Johnson. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// #include "cudaqs.h"
// #include <torch/extension.h>
// #include "../include/ctpl/ctpl_stl.h"

// #define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// std::vector<double> row_sum(torch::Tensor input){
//     CHECK_INPUT(input);
//     return row_sum_cuda(input);
// }

// std::vector<float> row_sum_ctpl(torch::Tensor input, int num_ctpl_threads){
//     CHECK_INPUT(input);
//     return row_sum_cuda_ctpl(input, num_ctpl_threads);
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("row_sum", &row_sum, "cudaqs row_sum");
//     m.def("row_sum_ctpl", &row_sum_ctpl, "cudaqs row_sum_ctpl");
// }

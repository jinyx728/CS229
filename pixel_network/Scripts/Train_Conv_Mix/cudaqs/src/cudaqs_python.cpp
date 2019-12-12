//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "cudaqs_global.h"
#include <torch/extension.h>
using namespace cudaqs;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<SpringData> spring_data(m,"SpringData");
    py::class_<AxialData> axial_data(m,"AxialData");
    py::class_<SpringSystem> spring_system(m,"SpringSystem");
    py::class_<NewtonOpt> newton_opt(m,"NewtonOpt");
    py::class_<BackwardOpt> backward_opt(m,"BackawrdOpt");
    py::class_<OptData,std::shared_ptr<OptData> > opt_data(m,"OptData");
    m.def("init", &init, "cudaqs init");
    m.def("init_spring", &init_spring, "cudaqs init_spring");
    m.def("init_axial", &init_axial, "cudaqs init_axial");
    m.def("init_system", &init_system, "cudaqs init_system");
    m.def("init_forward", &init_forward, "cudaqs init_forward");
    m.def("init_backward", &init_backward, "cudaqs init_backward");
    m.def("init_opt_data", &init_opt_data, "cudaqs init_opt_data");
    m.def("solve_forward", &solve_forward,"cudaqs solve_forward");
    m.def("solve_backward", &solve_backward,"cudaqs solve_backward");
}

//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "cvx_opt_global.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("init", &init,"init cvx opt"); 
  m.def("init_forward", &init_forward,"init forward opt");
  m.def("init_options", &init_options,"init options");
  m.def("solve_forward", &solve_forward,"solve forward opt");
  m.def("solve_forward_variable_m", &solve_forward_variable_m,"solve forward opt");
  m.def("init_backward", &init_backward,"init backward");
  m.def("solve_backward", &solve_backward,"solve backward");
  m.def("solve_backward_variable_m", &solve_backward_variable_m,"solve backward");
  m.def("init_lap",&init_lap,"init lap");
  m.def("init_spring",&init_spring,"init spring");
}
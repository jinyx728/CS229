//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include <iostream>
#include "ecos_opt.h"

void test()
{
    EcosOpt opt;
    
    std::vector<pfloat> m={1.0,1.0};
    std::vector<idxint> edges={0,1};
    std::vector<pfloat> l0={1.0};
    int n_vts=2;
    int n_edges=1;

    opt.init_solver(m,edges,l0,n_vts,n_edges);
    std::vector<pfloat> x;
    std::vector<pfloat> tgt_x={0.0,0.0,0.0,2.0,0.0,0.0};
    opt.solve(tgt_x,x);
    for(int i=0;i<n_vts*3;i++){
        std::cout<<x[i]<<" ";
    }

    tgt_x={0.0,0.0,0.0,0.0,2.0,0.0};
    opt.solve(tgt_x,x);
    for(int i=0;i<n_vts*3;i++){
        std::cout<<x[i]<<" ";
    }

    tgt_x={0.0,0.0,0.0,0.0,0.0,2.0};
    opt.solve(tgt_x,x);
    for(int i=0;i<n_vts*3;i++){
        std::cout<<x[i]<<" ";
    }
    std::cout<<std::endl;
}

int main()
{
    test();
}
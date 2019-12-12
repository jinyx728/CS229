//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#ifndef __ECOS_OPT_H__
#define __ECOS_OPT_H__
#include <ecos.h>
#include <memory>
#include <vector>

#define D 3

class EcosOpt{
public:
    EcosOpt();
    ~EcosOpt();
    int solve(bool verbose=false);
    void setup();
    void update();
    void cleanup();
    void convert_CCS(const std::vector<std::vector<pfloat> > &pr,const std::vector<std::vector<idxint> > &ir,std::vector<pfloat> &Xpr,std::vector<idxint> &Xjc,std::vector<idxint> &Xir) const;
    void make_work_copy();
    void init_options(int max_it=50,pfloat feastol=1e-8,pfloat abstol=1e-8,pfloat reltol=1e-8,pfloat ftol_inacc=1e-4,pfloat atol_inacc=5e-5,pfloat rtol_inacc=5e-5);
    void set_options(pwork *mwork);

    int n,m,p,l,ncones,nex;
    std::vector<idxint> q;
    std::vector<pfloat> c,h,b;
    std::vector<pfloat> Gpr;std::vector<idxint> Gjc,Gir;
    std::vector<pfloat> Apr;std::vector<idxint> Ajc,Air;

    std::vector<pfloat> c_work,h_work,b_work;
    std::vector<pfloat> Gpr_work;std::vector<idxint> Gjc_work,Gir_work;
    std::vector<pfloat> Apr_work;std::vector<idxint> Ajc_work,Air_work;

    pwork *mwork;
    int max_it;
    pfloat feastol,abstol,reltol,feastol_inacc,abstol_inacc,reltol_inacc;
};

template<class T>
void print(const std::vector<T> &vec);
#endif
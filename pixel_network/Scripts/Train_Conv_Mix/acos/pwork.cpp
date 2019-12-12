//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "pwork.h"
#include "solver.h"
#include <iostream>
#include <algorithm>
using namespace ACOS;

template<class LA>
void PWork<LA>::init(idxint _n, idxint _m, idxint _p, idxint _l, idxint _ncones, idxint* q,
                   pfloat* Gpr, idxint* Gjc, idxint* Gir,
                   pfloat* Apr, idxint* Ajc, idxint* Air,
                   pfloat* _c, pfloat* _h, pfloat* _b, pfloat *_x)
{
    n=_n;m=_m;p=_p;l=_l;ncones=_ncones;D=l+ncones;

    x.resize(n);y.resize(p);z.resize(m);s.resize(m);lambda.resize(m);
    dsaff_by_W.resize(m);dsaff.resize(m);dzaff.resize(m);saff.resize(m);zaff.resize(m);W_times_dzaff.resize(m);
    best_x.resize(n);best_y.resize(p);best_z.resize(m);best_s.resize(m);
    best_info=std::make_shared<Stats>();
    
    if(_x!=nullptr){
        std::copy(_x,_x+n,x.data());
        use_x0=true;
    }
    else{
        use_x0=false;
    }

    C=std::make_shared<Cone>();
    C->init(l,ncones,q);

    info=std::make_shared<Stats>();

    xequil.resize(n);Aequil.resize(p);Gequil.resize(m);

    init_stgs();

    c=LA::create_vec(_c,n);h=LA::create_vec(_h,m);b=LA::create_vec(_b,p);
    if(p>0){
        A=LA::create_mat(p,n,Apr,Ajc,Air);
    }
    G=LA::create_mat(m,n,Gpr,Gjc,Gir);

    kkt=std::make_shared<KKT>();

    rx.resize(n);ry.resize(p);rz.resize(m);
}

template<class LA>
void PWork<LA>::init_stgs()
{
    stgs=std::make_shared<Stgs>();
    stgs->maxit = MAXIT;
    stgs->gamma = GAMMA;
    stgs->delta = DELTA;
    stgs->eps = EPS;
    stgs->nitref = NITREF;
    stgs->abstol = ABSTOL;
    stgs->feastol = FEASTOL;
    stgs->reltol = RELTOL;
    stgs->abstol_inacc = ATOL_INACC;
    stgs->feastol_inacc = FTOL_INACC;
    stgs->reltol_inacc = RTOL_INACC;
    stgs->verbose = VERBOSE;
}

#include "la/la_eigen.h"
template class PWork<LAEigen>;
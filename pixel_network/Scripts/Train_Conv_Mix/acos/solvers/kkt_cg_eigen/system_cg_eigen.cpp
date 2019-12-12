//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "system_cg_eigen.h"
#include "pwork_cg_eigen.h"

using namespace ACOS;

template<class LA>
typename LA::Vec SystemCGEigen<LA>::multiply(const Vec &x) const
{
    return w->Gmt*W2_inv(w->Gm*x);
}

template<class LA>
pfloat SystemCGEigen<LA>::inner_product(const Vec &v1,const Vec &v2) const
{
    return v1.dot(v2);
}

template<class LA>
typename LA::Vec SystemCGEigen<LA>::precondition(const Vec &x) const
{
    // return Jacobi(w->Gcol_invsq,w->Gmt*W2(Jacobi(w->Grow_invsq,w->Gm*x)));
    // return w->Gmt*W2(w->Gm*x);
    return x;
}

template<class LA>
pfloat SystemCGEigen<LA>::convergence_norm(const Vec &r) const
{
    return LA::norminf(r);
}

template<class LA>
typename LA::Vec SystemCGEigen<LA>::W2_inv(const Vec &x) const
{
    if(isinit){
        return x;
    }
    Vec r(w->m);
    for(idxint i=0;i<w->C->lpc->p;i++){
        r[i]=x[i]*w->C->lpc->v_inv[i]; // v_inv=z/s
    }
    idxint x_offset=w->C->lpc->p;
    for(idxint i=0;i<w->C->nsoc;i++){
        idxint p=w->C->soc[i].p;
        r[x_offset]=w->C->soc[i].d0*x[x_offset];r.segment(x_offset+1,p-1)=x.segment(x_offset+1,p-1);
        Vec v;v.resize(p);
        v[0]=-w->C->soc[i].a;v.segment(1,p-1)=w->C->soc[i].q;
        r.segment(x_offset,p)+=v*(v.dot(x.segment(x_offset,p))*2);
        r.segment(x_offset,p)*=w->C->soc[i].eta_inv_square;
        x_offset+=p;
    }
    return r;
}

template<class LA>
typename LA::Vec SystemCGEigen<LA>::W2(const Vec &x) const
{
    if(isinit){
        return x;
    }
    Vec r(w->m);
    idxint j=0;
    for(idxint i=0;i<w->C->lpc->p;i++){
        r[i]=x[i]/w->C->lpc->v_inv[i]; // v_inv=z/s
    }
    idxint x_offset=w->C->lpc->p;
    for(idxint i=0;i<w->C->nsoc;i++){
        idxint p=w->C->soc[i].p;
        r[x_offset]=w->C->soc[i].d0*x[x_offset];r.segment(x_offset+1,p-1)=x.segment(x_offset+1,p-1);
        Vec v;v.resize(p);
        v[0]=w->C->soc[i].a;v.segment(1,p-1)=w->C->soc[i].q;
        r.segment(x_offset,p)+=v*(v.dot(x.segment(x_offset,p))*2);
        r.segment(x_offset,p)/=w->C->soc[i].eta_inv_square;
        x_offset+=p;
    }
    return r;
}

template<class LA>
typename LA::Vec SystemCGEigen<LA>::Jacobi(const Vec &v,const Vec &D) const
{
    return (v.array()*D.array()).matrix();
}

#include "la/la_eigen.h"
template class SystemCGEigen<LAEigen>;
//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include "glblopts.h"
#include <memory>
#include <vector>

namespace ACOS{
template<class LA>
class LPCone{
public:
    typedef typename LA::Vec Vec;
    idxint p;
    Vec w,v,v_inv;
    void init(idxint l){
        p=l;w.resize(l);v.resize(l);v_inv.resize(l);
    }
};

template<class LA>
class SOCone{
public:
    typedef typename LA::Vec Vec;
    idxint p;
    Vec skbar,zkbar;
    // used for ldl
    Vec q;
    pfloat a,eta;
    // used for cg
    pfloat eta_inv_square,d0;
    void init(idxint conesize){
        p=conesize;a=0;eta=0;
        q.resize(conesize-1);skbar.resize(conesize);zkbar.resize(conesize);
        eta_inv_square=0;d0=0;
    }
};

template<class LA>
class ConeData{
public:
    typedef std::shared_ptr<LPCone<LA> > LPConePtr;
    typedef std::shared_ptr<SOCone<LA> > SOConePtr;
    LPConePtr lpc;
    std::vector<SOCone<LA> > soc;
    idxint nsoc;
    void init(idxint l,idxint ncones,const idxint *q){
        lpc=std::make_shared<LPCone<LA> >();lpc->init(l);

        nsoc=ncones;
        if(ncones>0){
            soc.resize(ncones);
            for(int i=0;i<ncones;i++){
                idxint conesize=q[i];
                soc[i].init(conesize);
            }
        }
    }
};

};
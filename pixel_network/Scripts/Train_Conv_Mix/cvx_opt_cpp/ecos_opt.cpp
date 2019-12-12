//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "ecos_opt.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
EcosOpt::EcosOpt():n(0),m(0),p(0),l(0),ncones(0),nex(0),mwork(nullptr)
{
    init_options();
}

EcosOpt::~EcosOpt()
{
    if(mwork!=nullptr){
        ECOS_cleanup(mwork,0);
        mwork=nullptr;
    }
}

template<class T>
void print(const std::vector<T> &vec)
{
    for(auto iter=vec.begin();iter!=vec.end();iter++){
        std::cout<<*iter<<" ";
    }
    std::cout<<std::endl;
}

void EcosOpt::setup()
{
    make_work_copy();
    mwork=ECOS_setup(n,m,p,l,ncones,q.data(),nex,Gpr_work.data(),Gjc_work.data(),Gir_work.data(),Apr_work.data(),Ajc_work.data(),Air_work.data(),c_work.data(),h_work.data(),b_work.data(),0/*is_backward*/);
    if(mwork==nullptr){
        std::cout<<"EcosOpt:mwork==nullptr,something is wrong"<<std::endl;
    }
    // set_options(mwork);
}

void EcosOpt::update()
{
    for(int i=0;i<m;i++){
        ecos_updateDataEntry_h(mwork,i,h[i]);
    }
}

int EcosOpt::solve(bool verbose)
{
    // prevent ECOS from directly modifying our system matrix
    // make_work_copy();
    // mwork=ECOS_setup(n,m,p,l,ncones,q.data(),nex,Gpr_work.data(),Gjc_work.data(),Gir_work.data(),Apr_work.data(),Ajc_work.data(),Air_work.data(),c_work.data(),h_work.data(),b_work.data(),0/*is_backward*/);
    // if(mwork==nullptr){
    //     std::cout<<"mwork==nullptr,something is wrong"<<std::endl;
    // }
    set_options(mwork);
    update();
    mwork->stgs->verbose=verbose;
    int exitflag=ECOS_solve(mwork);
    // ECOS_cleanup(mwork,0);

    return exitflag;
}

void EcosOpt::init_options(int max_it,pfloat feastol,pfloat abstol,pfloat reltol,pfloat feastol_inacc,pfloat abstol_inacc,pfloat reltol_inacc)
{
    this->max_it=max_it;
    this->feastol=feastol;
    this->abstol=abstol;
    this->reltol=reltol;
    this->feastol_inacc=feastol_inacc;
    this->abstol_inacc=abstol_inacc;
    this->reltol_inacc=reltol_inacc;
}   

void EcosOpt::set_options(pwork *mwork)
{
    mwork->stgs->maxit=max_it;
    mwork->stgs->feastol=feastol;
    mwork->stgs->abstol=abstol;
    mwork->stgs->reltol=reltol;
    mwork->stgs->feastol_inacc=feastol_inacc;
    mwork->stgs->abstol_inacc=abstol_inacc;
    mwork->stgs->reltol_inacc=reltol_inacc;
}


void EcosOpt::cleanup()
{
    ECOS_cleanup(mwork,0);
    mwork=nullptr;
}

void EcosOpt::convert_CCS(const std::vector<std::vector<pfloat> > &pr,const std::vector<std::vector<idxint> > &ir,std::vector<pfloat> &Xpr,std::vector<idxint> &Xjc,std::vector<idxint> &Xir) const
{
    int total_entries=0;
    int n_cols=pr.size();
    for(int col_id=0;col_id<n_cols;col_id++){
        total_entries+=pr[col_id].size();
    }
    Xpr.resize(total_entries);
    Xjc.resize(n_cols+1);
    Xir.resize(total_entries);
    int processed_entries=0;
    Xjc[0]=0;
    for(int col_id=0;col_id<n_cols;col_id++){
        // assume pr[col_id].size()==ir[col_id].size()
        std::copy(pr[col_id].begin(),pr[col_id].end(),Xpr.begin()+processed_entries);
        std::copy(ir[col_id].begin(),ir[col_id].end(),Xir.begin()+processed_entries);
        processed_entries+=ir[col_id].size();
        Xjc[col_id+1]=processed_entries;
    }
    // Xjc[n_cols]=processed_entries;
}

void EcosOpt::make_work_copy()
{
    c_work=c;h_work=h;b_work=b;
    Gpr_work=Gpr;Gjc_work=Gjc;Gir_work=Gir;
    Apr_work=Apr;Ajc_work=Ajc;Air_work=Air;
}
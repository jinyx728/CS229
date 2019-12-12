//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "forward_opt.h"
#include <iostream>
#include <algorithm>
#include <cmath>

ForwardOpt::ForwardOpt():n_vts(0),n_edges(0),x_offset(0),t_offset(0),s_offset(0),l_offset(0)
{
}

ForwardOpt::~ForwardOpt()
{
}

void ForwardOpt::init_solver(const std::vector<pfloat> &w,const std::vector<idxint> &edges,const std::vector<pfloat> &l0,const int n_vts,const int n_edges)
{
    this->n_vts=n_vts;this->n_edges=n_edges;
    sqrt_w.resize(n_vts);
    for(int i=0;i<n_vts;i++){
        sqrt_w[i]=std::sqrt(double(w[i]));
    }

    n=n_vts*D+1+n_edges;
    m=1+D*n_vts+(D+1)*n_edges;
    p=n_edges;
    l=0;
    ncones=1+n_edges;
    nex=0;

    q.resize(ncones);
    q[0]=1+D*n_vts;
    std::fill(q.begin()+1,q.end(),(D+1));

    c.resize(n);
    std::fill(c.begin(),c.end(),0);
    c[n_vts*D]=1;

    x_offset=0;
    t_offset=n_vts*D;
    s_offset=n_vts*D+1;

    // this->edges=edges;

    create_G(edges,sqrt_w);
    create_h();
    create_A();
    create_b(l0);
    x0.resize(n);
    x0[n_vts*D]=0;
    std::copy(l0.begin(),l0.end(),x0.begin()+n_vts*D+1);
}

template<class Solver>
int ForwardOpt::solve(const std::vector<pfloat> &tgt_x,Solution &sol,bool verbose)
{
    for(int i=0;i<n_vts*D;i++){
        h[i+1]=tgt_x[i]*sqrt_w[i/D];
    }
    std::copy(tgt_x.begin(),tgt_x.end(),x0.begin());
    typename Solver::PWorkPtr mwork=Solver::setup(n,m,p,l,ncones,q.data(),nex,Gpr.data(),Gjc.data(),Gir.data(),Apr.data(),Ajc.data(),Air.data(),c.data(),h.data(),b.data(),x0.data());
    // mwork=Solver::setup(n,m,p,l,ncones,q.data(),nex,Gpr_work.data(),Gjc_work.data(),Gir_work.data(),Apr_work.data(),Ajc_work.data(),Air_work.data(),c_work.data(),h_work.data(),b_work.data());
    if(mwork==nullptr){
        std::cout<<"mwork==nullptr,something is wrong"<<std::endl;
    }
    mwork->stgs->verbose=verbose;

    int exitflag=Solver::solve(mwork);
    std::copy(mwork->x.data(),mwork->x.data()+n,sol.x.begin());
    std::copy(mwork->y.data(),mwork->y.data()+p,sol.y.begin());
    std::copy(mwork->z.data(),mwork->z.data()+m,sol.z.begin());
    std::copy(mwork->s.data(),mwork->s.data()+m,sol.s.begin());
    sol.success=1;
   
    Solver::cleanup(mwork);

    return exitflag;
}  
 
void ForwardOpt::create_G(const std::vector<idxint> &edges,const std::vector<pfloat> &sqrt_w)
{
    std::vector<std::vector<pfloat> > pr(n);
    std::vector<std::vector<idxint> > ir(n);
    idxint row_id=0;

    pr[t_offset].push_back(-1);
    ir[t_offset].push_back(row_id);
    row_id++;
    for(int vt_id=0;vt_id<n_vts;vt_id++){
        pfloat sqrt_wi=sqrt_w[vt_id];
        for(int i=0;i<D;i++){
            pr[x_offset+vt_id*D+i].push_back(sqrt_wi);
            ir[x_offset+vt_id*D+i].push_back(row_id);
            row_id++;
        }
    }

    for(int edge_id=0;edge_id<n_edges;edge_id++){
        idxint i0=edges[edge_id*2],i1=edges[edge_id*2+1];
        pr[s_offset+edge_id].push_back(-1);
        ir[s_offset+edge_id].push_back(row_id);
        row_id++;
        for(int i=0;i<D;i++){
            pr[x_offset+i0*D+i].push_back(1);
            ir[x_offset+i0*D+i].push_back(row_id);
            pr[x_offset+i1*D+i].push_back(-1);
            ir[x_offset+i1*D+i].push_back(row_id);
            row_id++;
        }
    }

    convert_CCS(pr,ir,Gpr,Gjc,Gir);
}

void ForwardOpt::create_h()
{
    h.resize(m);
    std::fill(h.begin(),h.end(),0);
}

void ForwardOpt::create_A()
{
    std::vector<std::vector<pfloat> > pr(n);
    std::vector<std::vector<idxint> > ir(n);
    idxint row_id=0;
    for(int edge_id=0;edge_id<n_edges;edge_id++){
        pr[s_offset+edge_id].push_back(1);
        ir[s_offset+edge_id].push_back(row_id);
        row_id++;
    }
    convert_CCS(pr,ir,Apr,Ajc,Air);
}

void ForwardOpt::create_b(const std::vector<pfloat> &l0)
{
    b.resize(n_edges);
    std::copy(l0.begin(),l0.end(),b.begin());
}

void ForwardOpt::convert_CCS(const std::vector<std::vector<pfloat> > &pr,const std::vector<std::vector<idxint> > &ir,std::vector<pfloat> &Xpr,std::vector<idxint> &Xjc,std::vector<idxint> &Xir) const
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
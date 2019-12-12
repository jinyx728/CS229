//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "backward_opt.h"
#include <iostream>
#include <algorithm>

BackwardOpt::BackwardOpt():n_vts(0),n_edges(0),use_spring(false),use_lap(false),forward_opt(nullptr)
{}

BackwardOpt::~BackwardOpt()
{}

void BackwardOpt::init_solver(ForwardOpt *_forward_opt)
{
    forward_opt=_forward_opt;
    n=forward_opt->n;m=forward_opt->m;p=forward_opt->p;l=forward_opt->l;ncones=forward_opt->ncones;nex=forward_opt->nex;
    q=forward_opt->q;c=forward_opt->c;h=forward_opt->h;b=forward_opt->b;
    // Gpr=forward_opt->Gpr;Gjc=forward_opt->Gjc;Gir=forward_opt->Gir;
    // Apr=forward_opt->Apr;Ajc=forward_opt->Ajc;Air=forward_opt->Air;
    n_edges=forward_opt->n_edges;n_vts=forward_opt->n_vts;
    use_spring=forward_opt->use_spring;
    use_lap=forward_opt->use_lap;
    sqrt_w=forward_opt->sqrt_w;
    Lt=forward_opt->L.transpose();

    setup();
}

void BackwardOpt::setup()
{
    Gpr_work=forward_opt->Gpr;Gjc_work=forward_opt->Gjc;Gir_work=forward_opt->Gir;
    Apr_work=forward_opt->Apr;Ajc_work=forward_opt->Ajc;Air_work=forward_opt->Air;
    c_work=forward_opt->c;h_work=forward_opt->h;b_work=forward_opt->b; // probably not useful
    mwork=ECOS_setup(n,m,p,l,ncones,q.data(),nex,Gpr_work.data(),Gjc_work.data(),Gir_work.data(),Apr_work.data(),Ajc_work.data(),Air_work.data(),c_work.data(),h_work.data(),b_work.data(),(idxint)GRAD_MODE/*is_backward*/);
    // printf("BackwardOpt::setup:h:%ld,h_work:%ld,forward_opt->h:%ld\n",(long int)mwork->h,(long int)h_work.data(),(long int)forward_opt->h.data());
    if(mwork==nullptr){
        std::cout<<"BackwardOpt: mwork==nullptr,something is wrong"<<std::endl;
        exit(-1);
    }
}

void BackwardOpt::update()
{
    for(int i=0;i<m;i++){
        ecos_updateDataEntry_h(mwork,i,forward_opt->h[i]); // dependency on the forward_opt, probably not the best idea ...
    }
}

idxint BackwardOpt::solve(const std::vector<pfloat> &tgt_x,const Solution &sol,const std::vector<pfloat> &in_grad,std::vector<pfloat> &out_grad)
{
    // make_work_copy();
    if (use_spring){
        forward_opt->update_G_h_spring(tgt_x,forward_opt->spring_id_to_G); // h does not need to be udated tho
        std::copy(forward_opt->sqrt_w.begin(),forward_opt->sqrt_w.end(),sqrt_w.begin());
        exit(-1);
    }
    // std::vector<pfloat> Gpr_work=forward_opt->Gpr;std::vector<idxint> Gjc_work=forward_opt->Gjc,Gir_work=forward_opt->Gir;
    // std::vector<pfloat> Apr_work=forward_opt->Apr;std::vector<idxint> Ajc_work=forward_opt->Ajc,Air_work=forward_opt->Air;
    // std::vector<pfloat> c_work=forward_opt->c,h_work=forward_opt->h,b_work=forward_opt->b; // probably not useful
    // mwork=ECOS_setup(n,m,p,l,ncones,q.data(),nex,Gpr_work.data(),Gjc_work.data(),Gir_work.data(),Apr_work.data(),Ajc_work.data(),Air_work.data(),c_work.data(),h_work.data(),b_work.data(),(idxint)GRAD_MODE/*is_backward*/);
    update();
    ECOS_backward_init(mwork,(idxint)GRAD_MODE);
    copy_and_prescale(mwork,sol.x,sol.y,sol.z,sol.s);
    std::vector<pfloat> RHSx;
    scale_and_get_RHSx(mwork,in_grad,RHSx);
    std::vector<pfloat> dx(n,0),dy(p,0),dz(m,0);
    idxint exitflag=ECOS_backward_prop(mwork,RHSx.data(),dx.data(),dy.data(),dz.data());
    backscale(mwork,dx,dy,dz);
    out_grad.resize(n_vts*D);
    std::fill(out_grad.begin(),out_grad.end(),0);   
    dh_dxj(dz,out_grad);
    if(use_spring){
        dG_dxj(sol.x,sol.z,dx,dz,tgt_x,out_grad);
    }

    // ECOS_cleanup(mwork,0);
    // mwork=nullptr;
    return exitflag;
}

idxint BackwardOpt::solve(const std::vector<pfloat> &tgt_x,const std::vector<pfloat> &w,const Solution &sol,const std::vector<pfloat> &in_grad,std::vector<pfloat> &out_grad,std::vector<pfloat> &out_m_grad)
{
    forward_opt->update_w_G(w);
    if (use_spring){
        forward_opt->update_G_h_spring(tgt_x,forward_opt->spring_id_to_G); // h does not need to be udated tho
    }
    std::vector<pfloat> Gpr_work=forward_opt->Gpr;std::vector<idxint> Gjc_work=forward_opt->Gjc,Gir_work=forward_opt->Gir;
    std::vector<pfloat> Apr_work=forward_opt->Apr;std::vector<idxint> Ajc_work=forward_opt->Ajc,Air_work=forward_opt->Air;
    std::vector<pfloat> c_work=forward_opt->c,h_work=forward_opt->h,b_work=forward_opt->b; // probably not useful
    mwork=ECOS_setup(n,m,p,l,ncones,q.data(),nex,Gpr_work.data(),Gjc_work.data(),Gir_work.data(),Apr_work.data(),Ajc_work.data(),Air_work.data(),c_work.data(),h_work.data(),b_work.data(),(idxint)GRAD_MODE/*is_backward*/);
    ECOS_backward_init(mwork,(idxint)GRAD_MODE);
    copy_and_prescale(mwork,sol.x,sol.y,sol.z,sol.s);
    std::vector<pfloat> RHSx;
    scale_and_get_RHSx(mwork,in_grad,RHSx);
    std::vector<pfloat> dx(n,0),dy(p,0),dz(m,0);
    idxint exitflag=ECOS_backward_prop(mwork,RHSx.data(),dx.data(),dy.data(),dz.data());
    backscale(mwork,dx,dy,dz);
    out_grad.resize(n_vts*D);
    std::fill(out_grad.begin(),out_grad.end(),0);   
    dh_dxj(dz,out_grad);
    if(use_spring){
        dG_dxj(sol.x,sol.z,dx,dz,tgt_x,out_grad);
    }
    out_m_grad.resize(n_vts);
    std::fill(out_m_grad.begin(),out_m_grad.end(),0);
    dh_dmj(dz,tgt_x,out_m_grad);
    dG_dmj(sol.x,sol.z,dx,dz,out_m_grad);
    for(int vt_id=0;vt_id<n_vts;vt_id++){
        out_m_grad[vt_id]*=0.5/sqrt_w[vt_id];
    }
    exit(-1);
    ECOS_cleanup(mwork,0);
    mwork=nullptr;
    return exitflag;
}

void BackwardOpt::copy_and_prescale(pwork *mwork,const std::vector<pfloat> &x,const std::vector<pfloat> &y,const std::vector<pfloat> &z,const std::vector<pfloat> &s)
{
    for(int i=0;i<n;i++) 
        mwork->x[i]=x[i]*mwork->xequil[i];
    for(int i=0;i<p;i++)
        mwork->y[i]=y[i]*mwork->Aequil[i];
    for(int i=0;i<m;i++)
        mwork->z[i]=z[i]*mwork->Gequil[i];
    for(int i=0;i<m;i++)
        mwork->s[i]=s[i]/mwork->Gequil[i];
}

void BackwardOpt::scale_and_get_RHSx(pwork *mwork,const std::vector<pfloat> &in_grad,std::vector<pfloat> &RHSx)
{
    RHSx.resize(n);std::fill(RHSx.begin(),RHSx.end(),0);
    for(int i=0;i<n_vts*D;i++){
        RHSx[i]=in_grad[i]/mwork->xequil[i];
    }
}

void BackwardOpt::backscale(const pwork *mwork,std::vector<pfloat> &dx,std::vector<pfloat> &dy,std::vector<pfloat> &dz)
{
    for(int i=0;i<mwork->n;i++){
        dx[i]/=mwork->xequil[i];
    }
    for(int i=0;i<mwork->p;i++){
        dy[i]/=mwork->Aequil[i];
    }
    for(int i=0;i<mwork->m;i++){
        dz[i]/=mwork->Gequil[i];
    }
}

void BackwardOpt::dh_dxj(const std::vector<pfloat> &dh,std::vector<pfloat> &dxj)
{
    for(int vt_id=0;vt_id<n_vts;vt_id++){
        for(int i=0;i<D;i++){
            dxj[vt_id*D+i]+=sqrt_w[vt_id]*dh[1+vt_id*D+i];
        }
    }
}

void BackwardOpt::dG_dxj(const std::vector<pfloat> &x,const std::vector<pfloat> &z,const std::vector<pfloat> &dx,const std::vector<pfloat> &dz,const std::vector<pfloat> &xj,std::vector<pfloat> &dxj)
{
    typedef Eigen::Map<Eigen::Matrix<pfloat,D,1> > Vec; 
    typedef Eigen::Map<const Eigen::Matrix<pfloat,D,1> > ConstVec; 
    const std::vector<idxint> &edges=forward_opt->edges;
    const std::vector<pfloat> &sqrt_stiffness=forward_opt->sqrt_stiffness;
    idxint spring_start=std::get<0>(forward_opt->spring_range),G_spring_start=std::get<0>(forward_opt->G_spring_range);
    for(int edge_id=0;edge_id<n_edges;edge_id++){
        idxint i0=edges[edge_id*2],i1=edges[edge_id*2+1];
        ConstVec xj0(xj.data()+i0*D,D),xj1(xj.data()+i1*D,D);
        Eigen::VectorXd dxj01=xj1-xj0;pfloat lj=dxj01.norm();
        Eigen::VectorXd ljhat=dxj01/lj;

        Vec dxj0(dxj.data()+i0*D,D),dxj1(dxj.data()+i1*D,D);
        
        ConstVec v(x.data()+spring_start+edge_id*D,D);
        Eigen::VectorXd t=(-ljhat*ljhat.dot(v)+v)/lj;
        t*=dz[G_spring_start+edge_id]*(-sqrt_stiffness[edge_id]);
        dxj1+=t;dxj0-=t;

        ConstVec dv(dx.data()+spring_start+edge_id*D,D);
        t=(-ljhat*ljhat.dot(dv)+dv)/lj;
        t*=z[G_spring_start+edge_id]*(-sqrt_stiffness[edge_id]);
        dxj1+=t;dxj0-=t;
    }
}

void BackwardOpt::dh_dmj(const std::vector<pfloat> &dh,const std::vector<pfloat> &tgt_x,std::vector<pfloat> &dmj)
{
    for(int vt_id=0;vt_id<n_vts;vt_id++){
        for(int i=0;i<D;i++){
            dmj[vt_id]+=dh[1+vt_id*D+i]*tgt_x[vt_id*D+i];
        }
    }
}

void BackwardOpt::dG_dmj(const std::vector<pfloat> &x,const std::vector<pfloat> &z,const std::vector<pfloat> &dx,const std::vector<pfloat> &dz,std::vector<pfloat> &dmj)
{
    for(int vt_id=0;vt_id<n_vts;vt_id++){
        for(int i=0;i<D;i++){
            dmj[vt_id]-=dz[1+vt_id*D+i]*x[vt_id*D+i];
            dmj[vt_id]-=dx[vt_id*D+i]*z[1+vt_id*D+i];
        }
    }
}

void BackwardOpt::scale_and_get_dx(pwork *mwork,const std::vector<pfloat> &dz,std::vector<pfloat> &dx)
{
    dx.resize(n_vts*D);std::fill(dx.begin(),dx.end(),0);
    for(int zi=1,xi=0;xi<n_vts*D;){
        int vt_id=xi/D;
        dx[xi]+=dz[zi]/mwork->Gequil[zi]*sqrt_w[vt_id];
        xi++;zi++;
    }
    if(use_lap){
        Eigen::VectorXd t(n_vts*D);
        idxint l_offset=1+n_vts*D+1;
        for(int ti=0,zi=l_offset;ti<n_vts*D;ti++,zi++){
            int vt_id=ti/D;
            t[ti]=dz[zi]/mwork->Gequil[zi]*sqrt_w[vt_id];
        }
        t=Lt*t;
        for(int i=0;i<n_vts*D;i++){
            dx[i]+=t[i];
        }
    }
}


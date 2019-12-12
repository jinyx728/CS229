//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "forward_opt.h"
#include <iostream>
#include <algorithm>
#include <cmath>


ForwardOpt::ForwardOpt():n_vts(0),n_edges(0),x_offset(0),t_offset(0),s_offset(0),l_offset(0),use_lap(false),pre_data(nullptr),relax_factor(1.01)
{
}

ForwardOpt::~ForwardOpt()
{
}

void ForwardOpt::init_solver(const std::vector<pfloat> &w,const std::vector<idxint> &edges,const std::vector<pfloat> &l0,const int n_vts,const int n_edges,PreinitData *pre_data)
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

    x_offset=0;
    t_offset=n_vts*D;
    s_offset=n_vts*D+1;

    this->pre_data=pre_data;
    this->use_spring=pre_data->use_spring;
    if(this->use_spring){
        std::get<0>(spring_range)=n;std::get<1>(spring_range)=n+n_vts*D;n+=n_edges*D;
        std::get<0>(G_spring_range)=1+D*n_vts;std::get<1>(G_spring_range)=1+D*n_vts+n_edges;m+=n_edges;
        std::get<0>(A_spring_range)=p;std::get<1>(A_spring_range)=p+D*n_edges;p+=D*n_edges;
        q[0]+=n_edges;
        dx_offset=std::get<0>(spring_range);
    }

    this->use_lap=pre_data->use_lap;
    if(this->use_lap){
        printf("lap untested\n");
        n+=1;
        m+=1+D*n_vts; // wrong if use spring
        ncones+=1;
        q.resize(ncones);
        q[1]=1+D*n_vts;q[ncones-1]=D+1;
        c[n_vts*D+1]=pre_data->lmd_lap;
        l_offset=n_vts*D+1;
        s_offset+=1;
        L.resize(n_vts*D,n_vts*D);
        create_L(pre_data->Lpr,pre_data->Ljc,pre_data->Lir,L);
    }

    c.resize(n);
    std::fill(c.begin(),c.end(),0);
    c[n_vts*D]=1;

    this->edges=edges;
    this->l0=l0;

    create_G(edges,sqrt_w);
    create_h();
    create_A();
    create_b(l0);
    compute_m_id_to_G();

    if(this->use_spring){
        get_id_to_G(G_spring_range,spring_id_to_G);
        get_stiffness(pre_data->youngs_modulus);
    }

    setup();
}

int ForwardOpt::solve(const std::vector<pfloat> &tgt_x,Solution &sol,bool verbose)
{
    for(int i=0;i<n_vts*D;i++){
        h[i+1]=tgt_x[i]*sqrt_w[i/D];
    }
    if(use_spring){
        update_G_h_spring(tgt_x,spring_id_to_G);
    }
    if(use_lap){
        update_h_lap(tgt_x);
    }

    int exitflag=EcosOpt::solve(verbose);
    check_sol(sol);
    std::copy(mwork->x,mwork->x+n,sol.x.begin());
    std::copy(mwork->y,mwork->y+p,sol.y.begin());
    std::copy(mwork->z,mwork->z+m,sol.z.begin());
    std::copy(mwork->s,mwork->s+m,sol.s.begin());
    if(exitflag==ECOS_OPTIMAL||exitflag==ECOS_INACC_OFFSET||exitflag==ECOS_MAXIT||exitflag==ECOS_NUMERICS){
        sol.success=1;
    }

    // check(x,tgt_x,lmd);
   
    // EcosOpt::cleanup();

    return exitflag;
}  

int ForwardOpt::solve(const std::vector<pfloat> &tgt_x,const std::vector<pfloat> &w,Solution &sol,bool verbose)
{
    update_w_G(w);
    return solve(tgt_x,sol,verbose);
}  

void ForwardOpt::check_sol(const Solution &sol) const
{
    if(sol.x.size()!=n){
        printf("ForwardOpt::check_sol,size mismatch,sol.x:%d,n:%d\n",(int)sol.x.size(),n);
    }
    if(sol.y.size()!=p){
        printf("ForwardOpt::check_sol,size mismatch,sol.y:%d,p:%d\n",(int)sol.y.size(),p);
    }
    if(sol.z.size()!=m){
        printf("ForwardOpt::check_sol,size mismatch,sol.z:%d,m:%d\n",(int)sol.z.size(),m);
    }
    if(sol.s.size()!=m){
        printf("ForwardOpt::check_sol,size mismatch,sol.s:%d,m:%d\n",(int)sol.z.size(),m);
    }
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

    if(use_spring){
        row_id=create_G_spring(row_id,pr,ir);
    }
    if(use_lap){
        row_id=create_G_lap(row_id,pr,ir);
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
    if(use_spring){
        row_id=create_A_spring(row_id,edges,pr,ir);
    }
    convert_CCS(pr,ir,Apr,Ajc,Air);
}

void ForwardOpt::create_b(const std::vector<pfloat> &l0)
{
    b.resize(p);
    std::copy(l0.begin(),l0.end(),b.begin());
    for(uint i=0;i<l0.size();i++){
        b[i]*=relax_factor;
    }
}

idxint ForwardOpt::create_G_spring(idxint row_id,std::vector<std::vector<pfloat> > &pr,std::vector<std::vector<idxint> > &ir)
{
    idxint spring_start=std::get<0>(spring_range);
    for(int edge_id=0;edge_id<n_edges;edge_id++){
        for(int i=0;i<D;i++){
            idxint xi=spring_start+edge_id*D+i;
            pr[xi].push_back(1);
            ir[xi].push_back(row_id);
        }
        row_id++;
    }
    return row_id;
}

idxint ForwardOpt::create_A_spring(idxint row_id,const std::vector<idxint> &edges,std::vector<std::vector<pfloat> > &pr,std::vector<std::vector<idxint> > &ir)
{
    idxint spring_start=std::get<0>(spring_range);
    for(int edge_id=0;edge_id<n_edges;edge_id++){
        idxint i0=edges[edge_id*2],i1=edges[edge_id*2+1];
        for(int i=0;i<D;i++){
            idxint dxi=spring_start+edge_id*D+i,x0i=i0*D+i,x1i=i1*D+i;
            pr[dxi].push_back(1);ir[dxi].push_back(row_id);
            pr[x0i].push_back(1);ir[x0i].push_back(row_id);
            pr[x1i].push_back(-1);ir[x1i].push_back(row_id);
            row_id++;
        }
    }
}

void ForwardOpt::update_w_G(const std::vector<pfloat> &w)
{
    for(int i=0;i<n_vts;i++){
        sqrt_w[i]=std::sqrt(double(w[i]));
    }
    for(int vt_id=0;vt_id<n_vts;vt_id++){
        for(int i=0;i<D;i++){
            idxint id=m_id_to_G[vt_id*D+i];
            Gpr[id]=sqrt_w[vt_id];
        }
    }
}

void ForwardOpt::update_G_h_spring(const std::vector<pfloat> &x,const std::vector<std::vector<idxint> > &spring_id_to_G)
{
    for(int edge_id=0;edge_id<n_edges;edge_id++){
        idxint i0=edges[edge_id*2],i1=edges[edge_id*2+1];
        Eigen::VectorXd x0=Eigen_From_Raw(x.data()+i0*D,D),x1=Eigen_From_Raw(x.data()+i1*D,D);
        Eigen::VectorXd dx=x1-x0;pfloat l=dx.norm();
        Eigen::VectorXd lhat=dx;
        if(l>1e-8){
            lhat/=l;
        }
        for(int i=0;i<D;i++){
            idxint xi=spring_id_to_G[edge_id][i];
            Gpr[xi]=-lhat[i]*sqrt_stiffness[edge_id];
        }
        idxint Gi=std::get<0>(G_spring_range)+edge_id;
        h[Gi]=(-l0[edge_id])*sqrt_stiffness[edge_id];
    }
}

void ForwardOpt::get_id_to_G(const range &r,std::vector<std::vector<idxint> > &id_to_G)
{
    int nrows=std::get<1>(r)-std::get<0>(r);
    id_to_G.clear();id_to_G.resize(nrows);
    for(uint col=0;col<Gjc.size()-1;col++){
        for(idxint j=Gjc[col];j<Gjc[col+1];j++){
            idxint row=Gir[j];
            if(row>=std::get<0>(r)&&row<std::get<1>(r)){
                id_to_G[row-std::get<0>(r)].push_back(j);
            }
        }
    }
}

idxint ForwardOpt::create_G_lap(idxint row_id,std::vector<std::vector<pfloat> > &pr,std::vector<std::vector<idxint> > &ir)
{
    printf("create_G_lap:untested\n");
    pr[l_offset].push_back(-1);
    ir[l_offset].push_back(row_id);
    row_id++;
    int p=0;
    int prev_rows=row_id;
    for(int vt_id=0;vt_id<n_vts;vt_id++){
        int next_p=pre_data->Ljc[vt_id];
        for(int i=0;i<D;i++){
            for(int j=p;j<next_p;j++){
                int vt_lap_id=pre_data->Lir[j];
                pr[x_offset+vt_id*D+i].push_back(pre_data->Lpr[j]*sqrt_w[vt_lap_id]);
                ir[x_offset+vt_id*D+i].push_back(pre_data->Lir[j]*D+i+prev_rows);
            }
        }
        p=next_p;
    }
    row_id+=n_vts*D; 
}

void ForwardOpt::update_h_lap(const std::vector<pfloat> &tgt_x)
{
    printf("update_h_lap:untested\n");
    int z_start=1+D*n_vts;
    idxint h_offset=1+n_vts*D+1;
    idxint tgt_offset=n_vts*D;
    z_start+=1+D*n_vts;
    Eigen::VectorXd x=Eigen_From_Raw(tgt_x.data(),n_vts*D);
    Eigen::VectorXd l=L*x;
    for(int i=0;i<n_vts*D;i++){
        h[h_offset+i]=l[i]*sqrt_w[i/D];
    }
}
 
void ForwardOpt::get_stiffness(const std::vector<pfloat> &youngs_modulus)
{
    sqrt_stiffness.resize(youngs_modulus.size());
    for(int edge_id=0;edge_id<youngs_modulus.size();edge_id++){
        sqrt_stiffness[edge_id]=std::sqrt(youngs_modulus[edge_id]/l0[edge_id]*pre_data->lmd_k);
    }
}

idxint ForwardOpt::find_ccs_id(const int row,const int col,const std::vector<idxint> &ir,const std::vector<idxint> &jc)
{
    for(idxint j=jc[col];j<jc[col+1];j++){
        if(ir[j]==row){
            return j;
        } 
    }
    return -1;
}

void ForwardOpt::compute_m_id_to_G()
{
    m_id_to_G.clear();
    for(int vt_id=0;vt_id<n_vts;vt_id++){
        for(int i=0;i<D;i++){
            idxint id=find_ccs_id(vt_id*D+i+1,vt_id*D+i,Gir,Gjc);
            if(id==-1){
                std::cout<<"compute_m_id_to_G(): id==-1,something is wrong!"<<std::endl;
            }
            m_id_to_G.push_back(id);
        }
    }
}

void create_ccs_mat(const int nrows,const int ncols,const std::vector<pfloat> &Lpr,const std::vector<idxint> &Ljc,const std::vector<idxint> &Lir,SpaMat &L)
{
    L.resize(nrows,ncols);
    typedef Eigen::Triplet<double> Triplet;
    std::vector<Triplet> triplets;
    for(int coli=0;coli<Ljc.size()-1;coli++){
        for(int pi=Ljc[coli];pi<Ljc[coli+1];pi++){
            double v=Lpr[pi];
            int row=Lir[pi];
            triplets.push_back(Triplet(row,coli,v));
        }
    }
    L.setFromTriplets(triplets.begin(),triplets.end());
}

Eigen::VectorXd Eigen_From_Raw(const pfloat *data,const int n)
{
    Eigen::VectorXd v(n);
    for(int i=0;i<n;i++){
        v[i]=data[i];
    }
    return v;
}

void create_L(const std::vector<pfloat> &Lpr,const std::vector<idxint> &Ljc,const std::vector<idxint> &Lir,SpaMat &L)
{
    typedef Eigen::Triplet<double> Triplet;
    std::vector<Triplet> triplets;
    int p=0;
    // the passed in L does not start from 0
    for(int coli=0;coli<Ljc.size();coli++){
        int next_p=Ljc[coli];
        for(int pi=p;pi<next_p;pi++){
            double v=Lpr[pi];
            int row=Lir[pi];
            for(int i=0;i<D;i++){
                triplets.push_back(Triplet(row*D+i,coli*D+i,v));
            }
        }
        p=next_p;
    }
    L.setFromTriplets(triplets.begin(),triplets.end());
}

void ForwardOpt::check(const std::vector<pfloat> &x_v,const std::vector<pfloat> &tgt_x_v,const std::vector<pfloat> &lmd_v)
{
    Eigen::VectorXd z=Eigen_From_Raw(mwork->z,m);
    Eigen::VectorXd s=Eigen_From_Raw(mwork->s,m);
    printf("create_ccs_mat:G\n");
    SpaMat G;create_ccs_mat(m,n,EcosOpt::Gpr,EcosOpt::Gjc,EcosOpt::Gir,G);
    Eigen::VectorXd dx=(G.transpose()*z).segment(0,n_vts*D);
    printf("G'z:norm:%.17g,max:%.17g\n",dx.norm(),dx.array().abs().matrix().maxCoeff());
    printf("create_ccs_mat:G_work\n");
    SpaMat G_work;create_ccs_mat(m,n,EcosOpt::Gpr_work,EcosOpt::Gjc_work,EcosOpt::Gir_work,G_work);
    dx=(G_work.transpose()*z).segment(0,n_vts*D);
    printf("Gwork'z:norm:%.17g,max:%.17g\n",dx.norm(),dx.array().abs().matrix().maxCoeff());

    printf("s0z0:%.17g,s.z:%.17g\n",s.dot(z),(s[0]*z.segment(1,n_vts*D)+z[0]*s.segment(1,n_vts*D)).norm());
    Eigen::VectorXd h=Eigen_From_Raw(EcosOpt::h.data(),m);
    Eigen::VectorXd x=Eigen_From_Raw(mwork->x,n);
    Eigen::VectorXd Ge=Eigen_From_Raw(mwork->Gequil,m);
    // Eigen::VectorXd gs=(s.array()/Ge.array()).matrix();
    Eigen::VectorXd rz=(h-G*x-s);
    printf("h-Gx-s:%.17g,%.17g\n",rz.norm(),(rz.array()/Ge.array()).matrix().norm());
    Eigen::VectorXd ze=(z.array()*Ge.array()).matrix();
    Eigen::VectorXd se=(s.array()/Ge.array()).matrix();
    double Ge_mean=Ge.segment(1,n_vts*D).mean();
    double Ge_var=(Ge.segment(1,n_vts*D).array()/(n_vts*D)*Ge.segment(1,n_vts*D).array()).matrix().sum()-Ge_mean*Ge_mean;
    printf("Ge:[0]:%.17g,mean:%.17g,std:%.17g\n",Ge[0],Ge_mean,Ge_var);
    printf("s0z0:%.17g,s.z:%.17g\n",se.segment(0,1+n_vts*D).dot(ze.segment(0,1+n_vts*D)),(se[0]*ze.segment(1,n_vts*D)+ze[0]*se.segment(1,n_vts*D)).norm());

    Eigen::VectorXd fullx=x;
    x=Eigen_From_Raw(x_v.data(),n_vts*D);
    Eigen::VectorXd tgt_x=Eigen_From_Raw(tgt_x_v.data(),n_vts*D);
    Eigen::VectorXd lmd=Eigen_From_Raw(lmd_v.data(),m);
    Eigen::VectorXd xdiff=x-tgt_x;
    double rt=lmd[0];
    Eigen::VectorXd rd=xdiff*rt;
    int edge_offset=1;
    if(use_lap){
        SpaMat L(n_vts*D,n_vts*D);create_L(pre_data->Lpr,pre_data->Ljc,pre_data->Lir,L);
        double rl=lmd[1];edge_offset+=1;
        rd+=L.transpose()*(L*xdiff)*rl;
    }
    for(int i=0;i<n_vts*D;i++){
        int vt_id=i/D;
        rd[i]*=sqrt_w[vt_id]*sqrt_w[vt_id];
    }
    for(int edgei=0;edgei<n_edges;edgei++){
        int i0=edges[edgei*2],i1=edges[edgei*2+1];
        Eigen::VectorXd v=x.segment(i1*D,D)-x.segment(i0*D,D);
        double ri=lmd[edge_offset+edgei];
        rd.segment(i0*D,D)+=-v*ri;
        rd.segment(i1*D,D)+=v*ri;
    }
    printf("rd:%.17g\n",rd.norm()); 
    Eigen::VectorXd sp=G*fullx-h;
    for(int mi=0,qi=0;qi<ncones;qi++){
        sp.segment(mi,q[qi])*=lmd[qi];
        mi+=q[qi];
    }
    printf("rd2:%.17g\n",(G.transpose()*sp).norm());
} 

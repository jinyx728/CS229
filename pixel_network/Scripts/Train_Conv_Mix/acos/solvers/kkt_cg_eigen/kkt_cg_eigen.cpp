//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "kkt_cg_eigen.h"
#include "pwork_cg_eigen.h"
#include "kkt_ldl_3x3.h"
#include "conjugate_gradient.h"
#include "impl/kkt.h"
#include <iostream>
#include <algorithm>
#include <assert.h>

using namespace ACOS;

template<class PWork>
void KKTCGEigen<PWork>::init(PWorkPtr w)
{
    idxint i;
    idxint nrhs;

    w->nm=w->n-w->p;

    KKTCGEigen<PWork>::set_equil(w);
    w->Gm=w->G.block(0,0,w->m,w->nm);w->Gm.makeCompressed();
    w->Gmt=w->Gm.transpose();w->Gmt.makeCompressed();
    w->Gs=w->G.block(0,w->nm,w->m,w->n-w->nm);w->Gs.makeCompressed();
    w->system.w=w;

    get_row_invsq(w->Gm,w->Grow_invsq);
    get_col_invsq(w->Gm,w->Gcol_invsq);

    if(w->p>0)
        w->At=LA::transpose(w->A);
    w->Gt=LA::transpose(w->G);

    // init cone
    w->C_impl = (kkt_ldl_3x3::cone *)MALLOC(sizeof(kkt_ldl_3x3::cone));

    /* LP cone */
    idxint l=w->C->lpc->p;
    w->C_impl->lpc = (kkt_ldl_3x3::lpcone *)MALLOC(sizeof(kkt_ldl_3x3::lpcone));
    w->C_impl->lpc->p = l;
    if( l > 0 ){
        w->C_impl->lpc->w = w->C->lpc->w.data();
        w->C_impl->lpc->v = w->C->lpc->v.data();
        w->C_impl->lpc->v_inv=w->C->lpc->v_inv.data();
        w->C_impl->lpc->kkt_idx = (idxint *)MALLOC(l*sizeof(idxint)); // init in createKKT_U
    } else {
        w->C_impl->lpc->w = NULL;
        w->C_impl->lpc->v = NULL;
        w->C_impl->lpc->kkt_idx = NULL;
    }


    /* Second-order cones */
    idxint ncones= w->C->nsoc;
    w->C_impl->soc = (w->C->nsoc == 0) ? NULL : (kkt_ldl_3x3::socone *)MALLOC(ncones*sizeof(kkt_ldl_3x3::socone));
    w->C_impl->nsoc = ncones;
    for( i=0; i<ncones; i++ ){
        idxint conesize = w->C->soc[i].p;
        w->C_impl->soc[i].p = conesize;
        w->C_impl->soc[i].a = 0;
        w->C_impl->soc[i].eta = 0;
        w->C_impl->soc[i].q = w->C->soc[i].q.data();
        w->C_impl->soc[i].skbar = w->C->soc[i].skbar.data();
        w->C_impl->soc[i].zkbar = w->C->soc[i].zkbar.data();
        w->C_impl->soc[i].Didx = (idxint *)MALLOC((conesize)*sizeof(idxint)); // init in createKKT_U
    }

    nrhs=w->n+w->p+w->m;
    w->nrhs=nrhs;
    w->dim_z=w->m;
    w->kkt->work1.resize(nrhs);
    w->kkt->rhs1.resize(nrhs);w->kkt->rhs2.resize(nrhs);
    w->kkt->dx1.resize(w->n);w->kkt->dx2.resize(w->n);
    w->kkt->dy1.resize(w->p);w->kkt->dy2.resize(w->p);
    w->kkt->dz1.resize(w->m);w->kkt->dz2.resize(w->m);
}

template<class PWork>
void KKTCGEigen<PWork>::init_rhs1(PWorkPtr w,Vec &rhs)
{
    idxint k = 0, j = 0;
    Vec rz_m_Gry;rz_m_Gry.resize(w->m);rz_m_Gry.setZero();
    rz_m_Gry=w->h-w->Gs*w->b;
    for( int i=0; i<w->nm; i++ ){ rhs[k++] = 0; }
    for( int i=0; i<w->C_impl->lpc->p; i++ ){ rhs[k++] = rz_m_Gry[i]; j++; }
    for( int l=0; l<w->C_impl->nsoc; l++ ){
        for( int i=0; i < w->C_impl->soc[l].p; i++ ){ rhs[k++] = rz_m_Gry[j++]; }
    }
    for( int i=0; i<w->p; i++ ){ rhs[k++] = w->b[i]; }
    for( int i=w->nm; i<w->n; i++ ){ rhs[k++] = 0; }
}

template<class PWork>
void KKTCGEigen<PWork>::init_rhs2(PWorkPtr w,Vec &rhs)
{
    idxint k=0;
    for( int i=0; i<w->nm; i++ ){ rhs[k++] = -w->c[i]; }
    for( int i=0; i<w->dim_z; i++) { rhs[k++] = 0; }
    for( int i=0; i<w->p; i++ ){ rhs[k++] = 0; }
    for( int i=w->nm; i<w->n; i++) { rhs[k++] = -w->c[i]; }
}

template<class PWork>
void KKTCGEigen<PWork>::update_rhs1(PWorkPtr w,Vec &rhs)
{
    for( int i=0; i<w->nm; i++){ rhs[i] = -w->c[i]; }
    for( int i=w->nm; i<w->n; i++ ){ rhs[i+w->dim_z+w->p] = -w->c[i]; }
}

template<class PWork>
typename KKTCGEigen<PWork>::KKTStat KKTCGEigen<PWork>::factor(PWorkPtr w)
{
    return KKTCGEigen<PWork>::KKT_OK;
}

template<class PWork>
void KKTCGEigen<PWork>::solve(PWorkPtr w,Vec &rhs,Vec &dx,Vec &dy,Vec &dz,idxint isinit)
{
    pfloat cg_tolerance=1e-6;
    idxint max_iters=w->m;
    Vec rhsz=w->Gmt*w->system.W2_inv(rhs.segment(w->nm,w->m))+rhs.segment(0,w->nm);
    Vec mx=dx.segment(0,w->nm);
    w->system.isinit=isinit;

    // Vec z0=rhs.segment(w->nm,w->m);
    // Vec z1=w->system.W2(w->system.W2_inv(z0));
    // printf("z1-z0:%.17f\n",LA::norminf(z1-z0));

    Vec r=rhsz-w->Gmt*w->system.W2_inv(w->Gm*mx);
    printf("r:%.17f\n",LA::norminf(r));
    
    idxint iters=ConjugateGradient<LA>::solve(w->system,mx,rhsz,cg_tolerance,max_iters);
    dx.segment(0,w->nm)=mx;
    if(w->stgs->verbose){
        printf("cg_iters:%d\n",iters);
    }
    dz=w->system.W2_inv(w->Gm*mx-rhs.segment(w->nm,w->m));
    idxint offset=w->nm+w->dim_z;
    for(int i=0;i<w->p;i++){ dx[w->nm+i]=rhs[offset+i]; }
    offset=w->nm+w->dim_z+w->p;
    for(int i=0;i<w->p;i++){ dy[i]=rhs[offset+i]; }
    dy-=w->Gs.transpose()*dz;
}

template<class PWork>
void KKTCGEigen<PWork>::update(PWorkPtr w,ConePtr Cp)
{
}

template<class PWork>
void KKTCGEigen<PWork>::RHS_affine(PWorkPtr w,Vec &rhs)
{
    idxint i, j, k, l;

    j = 0;
    Vec rz_m_Gry;rz_m_Gry.resize(w->m);rz_m_Gry.setZero();
    rz_m_Gry=(w->s-w->rz)-w->Gs*(-w->ry);
    for( i=0; i < w->nm; i++ ){ rhs[j++] = w->rx[i]; }
    for( i=0; i < w->C_impl->lpc->p; i++ ){ rhs[j++] = rz_m_Gry[i]; }
    k = w->C_impl->lpc->p;
    for( l=0; l < w->C_impl->nsoc; l++ ){
        for( i=0; i < w->C_impl->soc[l].p; i++ ){
            rhs[j++] = rz_m_Gry[k]; k++;
        }
    }
    for( i=0; i < w->p; i++ ){ rhs[j++] = -w->ry[i]; }
    for( i=w->nm; i< w->n; i++){ rhs[j++] = w->rx[i]; }
}

template<class PWork>
void KKTCGEigen<PWork>::RHS_combined(PWorkPtr w,Vec &rhs)
{
    Vec ds1(w->nrhs);
    Vec ds2(w->nrhs);
    idxint i, j, k, l;
    pfloat sigmamu = w->info->sigma * w->info->mu;
    pfloat one_minus_sigma = 1.0 - w->info->sigma;

    /* ds = lambda o lambda + W\s o Wz - sigma*mu*e) */
    kkt_ldl_3x3::conicProduct(w->lambda.data(), w->lambda.data(), w->C_impl, ds1.data());
    kkt_ldl_3x3::conicProduct(w->dsaff_by_W.data(), w->W_times_dzaff.data(), w->C_impl, ds2.data());
    for( i=0; i < w->C_impl->lpc->p; i++ ){ ds1[i] += ds2[i] - sigmamu; }
    k = w->C_impl->lpc->p;
    for( i=0; i < w->C_impl->nsoc; i++ ){
        ds1[k] += ds2[k] - sigmamu; k++;
        for( j=1; j < w->C_impl->soc[i].p; j++ ){ ds1[k] += ds2[k]; k++; }
    }

    /* dz = -(1-sigma)*rz + W*(lambda \ ds) */
    kkt_ldl_3x3::conicDivision(w->lambda.data(), ds1.data(), w->C_impl, w->dsaff_by_W.data());
    kkt_ldl_3x3::scale(w->dsaff_by_W.data(), w->C_impl, ds1.data());

    /* copy in RHS */
    Vec rz_m_Gry;rz_m_Gry.resize(w->m);rz_m_Gry.setZero();
    rz_m_Gry=-one_minus_sigma*w->rz+ds1.segment(0,w->m)-w->Gs*(-one_minus_sigma*w->ry);
    j = 0;
    for( i=0; i < w->nm; i++ ){ rhs[j++] *= one_minus_sigma; }
    for( i=0; i < w->C_impl->lpc->p; i++) { rhs[j++] = rz_m_Gry[i]; }
    k = w->C_impl->lpc->p;
    for( l=0; l < w->C_impl->nsoc; l++ ){
        for( i=0; i < w->C_impl->soc[l].p; i++ ){
            rhs[j++] = rz_m_Gry[k];
            k++;
        }
    }
    for( i=0; i < w->p; i++ ){ rhs[j++] *= one_minus_sigma; }
    for( i=w->nm; i<w->n; i++){ rhs[j++] *= one_minus_sigma; }
}

template<class PWork>
pfloat KKTCGEigen<PWork>::lineSearch(PWorkPtr w, Vec &lambda,Vec &ds, Vec &dz, pfloat tau, pfloat dtau, pfloat kap, pfloat dkap)
{
    return kkt_ldl_3x3::lineSearch(lambda.data(),ds.data(),dz.data(),tau,dtau,kap,dkap,w->C_impl,w->nrhs);
}

template<class PWork>
void KKTCGEigen<PWork>::cleanup(PWorkPtr w)
{
    /* Free memory for cones */
    if( w->C_impl->lpc->p > 0 ){
        FREE(w->C_impl->lpc->kkt_idx);
    }
    /* C->lpc is always allocated, so we free it here. */
    FREE(w->C_impl->lpc);

    for( idxint i=0; i < w->C_impl->nsoc; i++ ){
        FREE(w->C_impl->soc[i].Didx);
    }
    if( w->C_impl->nsoc > 0 ){
        FREE(w->C_impl->soc);
    }
    FREE(w->C_impl);
}

template<class PWork>
void KKTCGEigen<PWork>::scale(PWorkPtr w,Vec &z,Vec &lambda)
{
    kkt_ldl_3x3::scale(z.data(),w->C_impl,lambda.data()); // only lambda is updated
}

template<class PWork>
void KKTCGEigen<PWork>::bring2cone(PWorkPtr w,Vec &r,Vec &s)
{
    kkt_ldl_3x3::bring2cone(w->C_impl,r.data(),s.data()); // only s is updated
}

template<class PWork>
typename KKTCGEigen<PWork>::ConeStat KKTCGEigen<PWork>::updateScalings(PWorkPtr w,Vec &s,Vec &z,Vec &lambda)
{
    idxint stat=kkt_ldl_3x3::updateScalings(w->C_impl,s.data(),z.data(),lambda.data()); // C and lambda are updated
    for(idxint i=0;i<w->C_impl->nsoc;i++){
        w->C->soc[i].eta=w->C_impl->soc[i].eta;
        w->C->soc[i].a=w->C_impl->soc[i].a;
        w->C->soc[i].eta_inv_square=w->C_impl->soc[i].eta_inv_square;
        w->C->soc[i].d0=-1;
    }
    if(stat==1){
        return Base::OUTSIDE_CONE;
    }
    else{
        return Base::INSIDE_CONE;
    }
}

template<class PWork>
void KKTCGEigen<PWork>::set_equil(PWorkPtr w)
{
    idxint num_A_rows=w->p,num_G_rows=w->m;
    Vec xtmp(w->nm),Gtmp(num_G_rows);
    pfloat total;
    w->xequil.setOnes();w->Aequil.setOnes();w->Gequil.setOnes();
    Mat G=w->G.block(0,0,w->m,w->nm);G.makeCompressed();

    auto max_cols=[](const Mat &m,Vec &E) -> pfloat {
        for(idxint row=0;row<m.rows();row++){
            for(idxint j=m.outerIndexPtr()[row];j<m.outerIndexPtr()[row+1];j++){
                idxint col=m.innerIndexPtr()[j];
                pfloat v=m.valuePtr()[j];
                E[col]=MAX(fabs(v),E[col]);
            }
        }
    };
    auto max_rows=[](const Mat &m,Vec &E) -> pfloat {
        for(idxint row=0;row<m.rows();row++){
            for(idxint j=m.outerIndexPtr()[row];j<m.outerIndexPtr()[row+1];j++){
                idxint col=m.innerIndexPtr()[j];
                pfloat v=m.valuePtr()[j];
                E[row]=MAX(fabs(v),E[row]);
            }
        }
    };
    auto equil_cols=[](const Vec &E,Mat &m){
        for(idxint row=0;row<m.rows();row++){
            for(idxint j=m.outerIndexPtr()[row];j<m.outerIndexPtr()[row+1];j++){
                idxint col=m.innerIndexPtr()[j];
                m.valuePtr()[j]/=E[col];
            }
        }
    };
    auto equil_rows=[](const Vec &E,Mat &m){
        for(idxint row=0;row<m.rows();row++){
            for(idxint j=m.outerIndexPtr()[row];j<m.outerIndexPtr()[row+1];j++){
                idxint col=m.innerIndexPtr()[j];
                m.valuePtr()[j]/=E[col];
            }
        }
    };
    auto sqrt_vec=[](Vec &E){
        for(idxint i=0;i<E.size();i++){
            E[i]=fabs(E[i])<1e-10 ? 1.0 : sqrt(E[i]);
        }
    };
    auto mul_vec=[](Vec &a,const Vec &b){
        for(idxint i=0;i<a.size();i++){
            a[i]*=b[i];
        }
    };
    auto div_vec=[](Vec &a,const Vec &b){
        for(idxint i=0;i<a.size();i++){
            a[i]/=b[i];
        }
    };

    for(idxint iter=0;iter<EQUIL_ITERS;iter++){
        xtmp.setZero();Gtmp.setZero();
        if(num_G_rows>0){
            max_cols(G,xtmp);
            max_rows(G,Gtmp);
        }

        // cones
        // not sure what the right thing to do here
        idxint ind = w->C->lpc->p;
        for(idxint i = 0; i < w->C->nsoc; i++) {
            total = 0.0;
            for(idxint j = 0; j < w->C->soc[i].p; j++) {
                total += Gtmp[ind + j];
            }
            for(idxint j = 0; j < w->C->soc[i].p; j++) {
                Gtmp[ind + j] = total;
            }
            ind += w->C->soc[i].p;
        }

        sqrt_vec(xtmp);
        sqrt_vec(Gtmp);

        if(num_G_rows>0){
            equil_rows(Gtmp,G);
            equil_cols(xtmp,G);
        }

        for(idxint i=0;i<w->nm;i++){
            w->xequil[i]*=xtmp[i];
        }
        mul_vec(w->Gequil,Gtmp);
    }

    div_vec(w->h,w->Gequil);
    div_vec(w->c,w->xequil);
    std::copy(G.valuePtr(),G.valuePtr()+G.nonZeros(),w->G.valuePtr());
    // Gs=Ge-1Gs
    for(idxint row=w->nm;row<w->n;row++){
        pfloat sq=0;
        for(idxint j=w->G.outerIndexPtr()[row];j<w->G.outerIndexPtr()[row+1];j++){
            w->G.valuePtr()[j]/=w->Gequil[row];
        }
    }
}

template<class PWork>
void KKTCGEigen<PWork>::backscale(PWorkPtr w)
{
    idxint i;
    for( i=0; i < w->n; i++ ){ w->x[i] /= (w->xequil[i] * w->tau); }
    for( i=0; i < w->p; i++ ){ w->y[i] /= (w->tau); }
    for( i=0; i < w->m; i++ ){ w->z[i] /= (w->Gequil[i] * w->tau); }
    for( i=0; i < w->m; i++ ){ w->s[i] *= (w->Gequil[i] / w->tau); }
    for( i=0; i < w->n; i++ ){ w->c[i] *= w->xequil[i]; }
}

template<class PWork>
void KKTCGEigen<PWork>::get_row_invsq(const Mat &m,Vec &row_invsq)
{
    row_invsq.resize(m.rows());row_invsq.setZero();
    for(idxint row=0;row<m.rows();row++){
        for(idxint j=m.outerIndexPtr()[row];j<m.outerIndexPtr()[row+1];j++){
            idxint col=m.innerIndexPtr()[j];
            pfloat v=m.valuePtr()[j];
            row_invsq[row]+=v*v; // to be inverted later
        }
    }
    for(idxint row=0;row<m.rows();row++){
        if(row_invsq[row]>1e-10){
            row_invsq[row]=1/row_invsq[row];
        }
        else{
            row_invsq[row]=1;
        }
    }
}

template<class PWork>
void KKTCGEigen<PWork>::get_col_invsq(const Mat &m,Vec &col_invsq)
{
    col_invsq.resize(m.cols());col_invsq.setZero();
    for(idxint row=0;row<m.rows();row++){
        for(idxint j=m.outerIndexPtr()[row];j<m.outerIndexPtr()[row+1];j++){
            idxint col=m.innerIndexPtr()[j];
            pfloat v=m.valuePtr()[j];
            col_invsq[col]+=v*v;// to be inverted later
        }
    }
    for(idxint col=0;col<col_invsq.size();col++){
        if(col_invsq[col]>1e-10){
            col_invsq[col]=1/col_invsq[col];
        }
        else{
            col_invsq[col]=1;
        }
    }
}

#include "la/la_eigen.h"
template class KKTCGEigen<PWorkCGEigen<LAEigen> >;
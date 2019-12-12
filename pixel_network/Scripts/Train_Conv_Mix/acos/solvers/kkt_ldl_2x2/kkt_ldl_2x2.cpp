//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "kkt_ldl_2x2.h"
#include "pwork_ldl_2x2.h"
#include "impl/spla.h"
#include "impl/splamm.h"
/* MATRIX ORDERING LIBRARY --------------------------------------------- */
#include "amd.h"
#include "amd_internal.h"
#include <iostream>
#include <algorithm>

/* SPARSE LDL LIBRARY -------------------------------------------------- */
#ifdef __cplusplus
extern "C"{
#endif
#include "ldl.h"
#ifdef __cplusplus
}
#endif

using namespace ACOS;

template<class LA>
static kkt_ldl_3x3::spmat* get_spmat(typename LA::Mat &m) // free mat!!!!!!!!!
{
    kkt_ldl_3x3::spmat* M=(kkt_ldl_3x3::spmat*)MALLOC(sizeof(kkt_ldl_3x3::spmat));
    M->m=m.rows();
    M->n=m.cols();
    M->nnz=m.nonZeros();
    M->jc=(idxint*)MALLOC(sizeof(idxint)*(M->n+1));
    std::copy(m.outerIndexPtr(),m.outerIndexPtr()+(M->n+1),M->jc);
    M->ir=(idxint*)MALLOC(sizeof(idxint)*M->nnz);
    std::copy(m.innerIndexPtr(),m.innerIndexPtr()+M->nnz,M->ir);
    M->pr=(pfloat*)MALLOC(sizeof(pfloat)*M->nnz);
    std::copy(m.valuePtr(),m.valuePtr()+M->nnz,M->pr);
    return M;
}

// template<class PWork>
// static std::vector<typename PWork::tuple> get_tuplets(typename PWork::Mat &m) // assuming colume major
// {
//     std::vector<typename PWork::tuple> tuples(m.nonZeros());
//     for(idxint col=0;col<m.cols;col++){
//         for(idxint j=m.outerIndexPtr()[col];j<m.outerIndexPtr()[col+1];j++){
//             idxint row=m.innerIndexPtr()[j];
//             pfloat v=m.valuePtr()[j];
//             tuples[j]=PWork::tuple(row,col,v);
//         }
//     }
//     return tuples;
// }  

template<class PWork>
void KKTLDL2x2<PWork>::init(PWorkPtr w)
{
    idxint i, cidx, conesize, lnz, amd_result, nK, *Ljc, *Lir, *P, *Pinv, *Sign;
    mfloat Control [AMD_CONTROL], Info [AMD_INFO];
    pfloat *Lpr;
    kkt_ldl_3x3::spmat *At, *Gt, *KU;
    idxint *AtoAt, *GtoGt, *AttoK, *GttoK;
    idxint nrhs;

    w->nm=w->n-w->p;

    KKTLDL2x2<PWork>::set_equil(w);

    w->A_impl=nullptr;
    typename LA::Mat G_impl=w->G.block(0,0,w->m,w->nm);G_impl.makeCompressed();
    w->G_impl=get_spmat<LA>(G_impl);
    typename LA::Mat Gs=w->G.block(0,w->nm,w->m,w->n-w->nm);Gs.makeCompressed();
    w->Gs=Gs;
    // typename LA::Mat As_inv=w->A.block(0,w->nm,w->p,w->n-w->nm);As_inv.makeCompressed();
    // for(i=0;i<As_inv.nonZeros();i++){ //inverse
    //     As_inv.valuePtr()[i]=1/As_inv.valuePtr()[i];
    // }
    // w->GsAs_inv=Gs*As_inv;

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
    cidx = 0;
    for( i=0; i<ncones; i++ ){
        conesize = w->C->soc[i].p;
        w->C_impl->soc[i].p = conesize;
        w->C_impl->soc[i].a = 0;
        w->C_impl->soc[i].eta = 0;
        w->C_impl->soc[i].q = w->C->soc[i].q.data();
        w->C_impl->soc[i].skbar = w->C->soc[i].skbar.data();
        w->C_impl->soc[i].zkbar = w->C->soc[i].zkbar.data();
        w->C_impl->soc[i].Didx = (idxint *)MALLOC((conesize)*sizeof(idxint)); // init in createKKT_U
        cidx += conesize;
    }

    // init kkt
    GtoGt=(idxint *)MALLOC(w->G.nonZeros()*sizeof(idxint));
    GttoK = (idxint *)MALLOC(w->G.nonZeros()*sizeof(idxint));
    Gt=kkt_ldl_3x3::transposeSparseMatrix(w->G_impl,GtoGt);
    At=nullptr;AtoAt=nullptr;AttoK=nullptr;
    kkt_ldl_3x3::createKKT_U(Gt,At,w->C_impl,&Sign, &KU, AttoK, GttoK);

    // if(w->p>0){
    //     w->AtoK=(idxint *)MALLOC(w->A.nonZeros()*sizeof(idxint));
    //     for(i=0; i<w->A.nonZeros(); i++){ w->AtoK[i] = AttoK[AtoAt[i]]; }
    // }
    // w->GtoK=(idxint *)MALLOC(w->G.nonZeros()*sizeof(idxint));
    // for(i=0; i<w->G.nonZeros(); i++){ w->GtoK[i] = GttoK[GtoGt[i]]; }
    nK = KU->n;
    nrhs=w->n+w->p+w->m+2*w->C->nsoc;
    w->dim_z=w->m+2*w->C->nsoc;
    w->nK=nK;

    w->kkt->work1.resize(nrhs);
    w->kkt->rhs1.resize(nrhs);w->kkt->rhs2.resize(nrhs);
    w->kkt->dx1.resize(w->n);w->kkt->dx2.resize(w->n); w->kkt->dx1.setZero();
    w->kkt->dy1.resize(w->p);w->kkt->dy2.resize(w->p);
    w->kkt->dz1.resize(w->m);w->kkt->dz2.resize(w->m);

    w->kkt_impl = (kkt_ldl_3x3::kkt *)MALLOC(sizeof(kkt_ldl_3x3::kkt));
    w->kkt_impl->D = (pfloat *)MALLOC(nK*sizeof(pfloat));
    w->kkt_impl->Parent = (idxint *)MALLOC(nK*sizeof(idxint));
    w->kkt_impl->Pinv = (idxint *)MALLOC(nK*sizeof(idxint));
    w->kkt_impl->work1 = w->kkt->work1.data();
    w->kkt_impl->work2 = (pfloat *)MALLOC(nK*sizeof(pfloat));
    w->kkt_impl->work3 = (pfloat *)MALLOC(nK*sizeof(pfloat));
    w->kkt_impl->work4 = (pfloat *)MALLOC(nK*sizeof(pfloat));
    w->kkt_impl->work5 = (pfloat *)MALLOC(nK*sizeof(pfloat));
    w->kkt_impl->work6 = (pfloat *)MALLOC(nK*sizeof(pfloat));
    w->kkt_impl->Flag = (idxint *)MALLOC(nK*sizeof(idxint));
    w->kkt_impl->Pattern = (idxint *)MALLOC(nK*sizeof(idxint));
    w->kkt_impl->Lnz = (idxint *)MALLOC(nK*sizeof(idxint));
    w->kkt_impl->RHS1 = w->kkt->rhs1.data();
    w->kkt_impl->RHS2 = w->kkt->rhs2.data();
    w->kkt_impl->dx1 = w->kkt->dx1.data();
    w->kkt_impl->dx2 = w->kkt->dx2.data();
    w->kkt_impl->dy1 = w->kkt->dy1.data();
    w->kkt_impl->dy2 = w->kkt->dy2.data();
    w->kkt_impl->dz1 = w->kkt->dz1.data();
    w->kkt_impl->dz2 = w->kkt->dz2.data();
    w->kkt_impl->Sign = (idxint *)MALLOC(nK*sizeof(idxint));
    w->kkt_impl->PKPt = kkt_ldl_3x3::newSparseMatrix(nK, nK, KU->nnz);
    w->kkt_impl->PK = (idxint *)MALLOC(KU->nnz*sizeof(idxint));
    
    P = (idxint *)MALLOC(nK*sizeof(idxint));
    AMD_defaults(Control);
    amd_result = AMD_order(nK, KU->jc, KU->ir, P, Control, Info);

    if( amd_result != AMD_OK ){
        PRINTTEXT("Problem in AMD ordering, exiting.\n");
        AMD_info(Info);
        return;
    }

    kkt_ldl_3x3::pinv(nK, P, w->kkt_impl->Pinv);
    Pinv = w->kkt_impl->Pinv;

    kkt_ldl_3x3::permuteSparseSymmetricMatrix(KU, w->kkt_impl->Pinv, w->kkt_impl->PKPt, w->kkt_impl->PK);
    for( i=0; i<nK; i++ ){ w->kkt_impl->Sign[Pinv[i]] = Sign[i]; }

    /* symbolic factorization */
    Ljc = (idxint *)MALLOC((nK+1)*sizeof(idxint));
    LDL_symbolic2(
        w->kkt_impl->PKPt->n,    /* A and L are n-by-n, where n >= 0 */
        w->kkt_impl->PKPt->jc,   /* input of size n+1, not modified */
        w->kkt_impl->PKPt->ir,   /* input of size nz=Ap[n], not modified */
        Ljc,                     /* output of size n+1, not defined on input */
        w->kkt_impl->Parent,     /* output of size n, not defined on input */
        w->kkt_impl->Lnz,        /* output of size n, not defined on input */
        w->kkt_impl->Flag        /* workspace of size n, not defn. on input or output */
    );


    /* assign memory for L */
    lnz = Ljc[nK];
    Lir = (idxint *)MALLOC(lnz*sizeof(idxint));
    Lpr = (pfloat *)MALLOC(lnz*sizeof(pfloat));
    w->kkt_impl->L = kkt_ldl_3x3::ecoscreateSparseMatrix(nK, nK, lnz, Ljc, Lir, Lpr);

    /* permute kkt_impl matrix - we work on this one from now on */
    kkt_ldl_3x3::permuteSparseSymmetricMatrix(KU, w->kkt_impl->Pinv, w->kkt_impl->PKPt, NULL);

    /* clean up */
    w->kkt_impl->P = P;

    FREE(Sign);
    if(At) {
        kkt_ldl_3x3::freeSparseMatrix(At);
        FREE(AtoAt);
        FREE(AttoK);
    }
    kkt_ldl_3x3::freeSparseMatrix(Gt);
    kkt_ldl_3x3::freeSparseMatrix(KU);
    FREE(GtoGt);
    FREE(GttoK);
}

template<class PWork>
void KKTLDL2x2<PWork>::init_rhs1(PWorkPtr w,Vec &rhs)
{
    idxint k = 0, j = 0;
    Vec rz_m_Gry;rz_m_Gry.resize(w->m);rz_m_Gry.setZero();
    rz_m_Gry=w->h-w->Gs*w->b;
    for( int i=0; i<w->nm; i++ ){ rhs[w->kkt_impl->Pinv[k++]] = 0; }
    for( int i=0; i<w->C_impl->lpc->p; i++ ){ rhs[w->kkt_impl->Pinv[k++]] = rz_m_Gry[i]; j++; }
    for( int l=0; l<w->C_impl->nsoc; l++ ){
        for( int i=0; i < w->C_impl->soc[l].p; i++ ){ rhs[w->kkt_impl->Pinv[k++]] = rz_m_Gry[j++]; }
        rhs[w->kkt_impl->Pinv[k++]] = 0;
        rhs[w->kkt_impl->Pinv[k++]] = 0;
    }
    for( int i=0; i<w->p; i++ ){ rhs[k++] = w->b[i]; }
    for( int i=w->nm; i<w->n; i++ ){ rhs[k++] = 0; }
}

template<class PWork>
void KKTLDL2x2<PWork>::init_rhs2(PWorkPtr w,Vec &rhs)
{
    idxint k=0;
    for( int i=0; i<w->nm; i++ ){ rhs[w->kkt_impl->Pinv[k++]] = -w->c[i]; }
    for( int i=0; i<w->dim_z; i++) { rhs[w->kkt_impl->Pinv[k++]] = 0; }
    for( int i=0; i<w->p; i++ ){ rhs[k++] = 0; }
    for( int i=w->nm; i<w->n; i++) { rhs[k++] = -w->c[i]; }
}

template<class PWork>
void KKTLDL2x2<PWork>::update_rhs1(PWorkPtr w,Vec &rhs)
{
    for( int i=0; i<w->nm; i++){ rhs[w->kkt_impl->Pinv[i]] = -w->c[i]; }
    for( int i=w->nm; i<w->n; i++ ){ rhs[i+w->dim_z+w->p] = -w->c[i]; }
}

template<class PWork>
typename KKTLDL2x2<PWork>::KKTStat KKTLDL2x2<PWork>::factor(PWorkPtr w)
{
    idxint r=kkt_ldl_3x3::kkt_factor(w->kkt_impl,w->stgs->eps,w->stgs->delta); // only kkt_impl is updated 
    if(r==1){
        return Base::KKT_OK;
    }
    else{
        return Base::KKT_PROBLEM;
    }
}

template<class PWork>
void KKTLDL2x2<PWork>::solve(PWorkPtr w,Vec &rhs,Vec &dx,Vec &dy,Vec &dz,idxint isinit)
{
    kkt_ldl_3x3::kkt_solve(w->kkt_impl, nullptr, w->G_impl, rhs.data(), dx.data(), nullptr, dz.data(), w->nm, 0, w->m, w->C_impl, isinit, w->stgs->nitref); // only dx,dy,dz is updated
    idxint offset=w->nm+w->dim_z;
    for(int i=0;i<w->p;i++){ dx[w->nm+i]=rhs[offset+i]; }
    offset=w->nm+w->dim_z+w->p;
    for(int i=0;i<w->p;i++){ dy[i]=rhs[offset+i]; }
    dy-=w->Gs.transpose()*dz;
}

template<class PWork>
void KKTLDL2x2<PWork>::update(PWorkPtr w,ConePtr Cp)
{
    kkt_ldl_3x3::kkt_update(w->kkt_impl->PKPt, w->kkt_impl->PK, w->C_impl); // #only PKPt is updated
}

template<class PWork>
void KKTLDL2x2<PWork>::RHS_affine(PWorkPtr w,Vec &rhs)
{
    idxint i, j, k, l;
    idxint* Pinv = w->kkt_impl->Pinv;

    j = 0;
    Vec rz_m_Gry;rz_m_Gry.resize(w->m);rz_m_Gry.setZero();
    rz_m_Gry=(w->s-w->rz)-w->Gs*(-w->ry);
    for( i=0; i < w->nm; i++ ){ rhs[Pinv[j++]] = w->rx[i]; }
    for( i=0; i < w->C_impl->lpc->p; i++ ){ rhs[Pinv[j++]] = rz_m_Gry[i]; }
    k = w->C_impl->lpc->p;
    for( l=0; l < w->C_impl->nsoc; l++ ){
        for( i=0; i < w->C_impl->soc[l].p; i++ ){
            rhs[Pinv[j++]] = rz_m_Gry[k]; k++;
        }
        rhs[Pinv[j++]] = 0;
        rhs[Pinv[j++]] = 0;
    }
    for( i=0; i < w->p; i++ ){ rhs[j++] = -w->ry[i]; }
    for( i=w->nm; i< w->n; i++){ rhs[j++] = w->rx[i]; }
}

template<class PWork>
void KKTLDL2x2<PWork>::RHS_combined(PWorkPtr w,Vec &rhs)
{
    pfloat* ds1 = w->kkt_impl->work1;
    pfloat* ds2 = w->kkt_impl->work2;
    idxint i, j, k, l;
    pfloat sigmamu = w->info->sigma * w->info->mu;
    pfloat one_minus_sigma = 1.0 - w->info->sigma;
    idxint* Pinv = w->kkt_impl->Pinv;


    /* ds = lambda o lambda + W\s o Wz - sigma*mu*e) */
    kkt_ldl_3x3::conicProduct(w->lambda.data(), w->lambda.data(), w->C_impl, ds1);
    kkt_ldl_3x3::conicProduct(w->dsaff_by_W.data(), w->W_times_dzaff.data(), w->C_impl, ds2);
    for( i=0; i < w->C_impl->lpc->p; i++ ){ ds1[i] += ds2[i] - sigmamu; }
    k = w->C_impl->lpc->p;
    for( i=0; i < w->C_impl->nsoc; i++ ){
        ds1[k] += ds2[k] - sigmamu; k++;
        for( j=1; j < w->C_impl->soc[i].p; j++ ){ ds1[k] += ds2[k]; k++; }
    }

    /* dz = -(1-sigma)*rz + W*(lambda \ ds) */
    kkt_ldl_3x3::conicDivision(w->lambda.data(), ds1, w->C_impl, w->dsaff_by_W.data());
    kkt_ldl_3x3::scale(w->dsaff_by_W.data(), w->C_impl, ds1);

    /* copy in RHS */
    Vec rz_m_Gry;rz_m_Gry.resize(w->m);rz_m_Gry.setZero();
    rz_m_Gry=-one_minus_sigma*w->rz+w->kkt->work1.segment(0,w->m)-w->Gs*(-one_minus_sigma*w->ry); // use wor1 for ds1
    j = 0;
    for( i=0; i < w->nm; i++ ){ rhs[Pinv[j++]] *= one_minus_sigma; }
    for( i=0; i < w->C_impl->lpc->p; i++) { rhs[Pinv[j++]] = rz_m_Gry[i]; }
    k = w->C_impl->lpc->p;
    for( l=0; l < w->C_impl->nsoc; l++ ){
        for( i=0; i < w->C_impl->soc[l].p; i++ ){
            rhs[Pinv[j++]] = rz_m_Gry[k];
            k++;
        }
        rhs[Pinv[j++]] = 0;
        rhs[Pinv[j++]] = 0;
    }
    for( i=0; i < w->p; i++ ){ rhs[j++] *= one_minus_sigma; }
    for( i=w->nm; i<w->n; i++){ rhs[j++] *= one_minus_sigma; }
}

template<class PWork>
pfloat KKTLDL2x2<PWork>::lineSearch(PWorkPtr w, Vec &lambda,Vec &ds, Vec &dz, pfloat tau, pfloat dtau, pfloat kap, pfloat dkap)
{
    return kkt_ldl_3x3::lineSearch(lambda.data(),ds.data(),dz.data(),tau,dtau,kap,dkap,w->C_impl,w->kkt_impl);
}

template<class PWork>
void KKTLDL2x2<PWork>::cleanup(PWorkPtr w)
{
    FREE(w->kkt_impl->D);               
    FREE(w->kkt_impl->Flag);            
    kkt_ldl_3x3::freeSparseMatrix(w->kkt_impl->L);
    FREE(w->kkt_impl->Lnz);              
    FREE(w->kkt_impl->Parent);           
    FREE(w->kkt_impl->Pattern);          
    FREE(w->kkt_impl->Sign);             
    FREE(w->kkt_impl->Pinv);             
    FREE(w->kkt_impl->P);
    FREE(w->kkt_impl->PK);               
    kkt_ldl_3x3::freeSparseMatrix(w->kkt_impl->PKPt);
    FREE(w->kkt_impl->work2);      
    FREE(w->kkt_impl->work3);      
    FREE(w->kkt_impl->work4);      
    FREE(w->kkt_impl->work5);      
    FREE(w->kkt_impl->work6);      
    FREE(w->kkt_impl);             
    // if (w->p>0) {
    //     FREE(w->AtoK);
    // }
    // FREE(w->GtoK);

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

    if(w->A_impl)
        kkt_ldl_3x3::freeSparseMatrix(w->A_impl);
    kkt_ldl_3x3::freeSparseMatrix(w->G_impl);
}

template<class PWork>
void KKTLDL2x2<PWork>::scale(PWorkPtr w,Vec &z,Vec &lambda)
{
    kkt_ldl_3x3::scale(z.data(),w->C_impl,lambda.data()); // only lambda is updated
}

template<class PWork>
void KKTLDL2x2<PWork>::bring2cone(PWorkPtr w,Vec &r,Vec &s)
{
    kkt_ldl_3x3::bring2cone(w->C_impl,r.data(),s.data()); // only s is updated
}

template<class PWork>
typename KKTLDL2x2<PWork>::ConeStat KKTLDL2x2<PWork>::updateScalings(PWorkPtr w,Vec &s,Vec &z,Vec &lambda)
{
    idxint stat=kkt_ldl_3x3::updateScalings(w->C_impl,s.data(),z.data(),lambda.data()); // C and lambda are updated
    for(idxint i=0;i<w->C_impl->nsoc;i++){
        w->C->soc[i].eta=w->C_impl->soc[i].eta;
        w->C->soc[i].a=w->C_impl->soc[i].a;
    }
    if(stat==1){
        return Base::OUTSIDE_CONE;
    }
    else{
        return Base::INSIDE_CONE;
    }
}

template<class PWork>
void KKTLDL2x2<PWork>::set_equil(PWorkPtr w)
{
    idxint num_A_rows=w->p,num_G_rows=w->m;
    Vec xtmp(w->nm),Gtmp(num_G_rows);
    pfloat total;
    w->xequil.setOnes();w->Aequil.setOnes();w->Gequil.setOnes();
    Mat G=w->G.block(0,0,w->m,w->nm);

    auto max_cols=[](const Mat &m,Vec &E) -> pfloat {
        for(idxint col=0;col<m.cols();col++){
            for(idxint j=m.outerIndexPtr()[col];j<m.outerIndexPtr()[col+1];j++){
                idxint row=m.innerIndexPtr()[j];
                pfloat v=m.valuePtr()[j];
                E[col]=MAX(fabs(v),E[col]);
            }
        }
    };
    auto max_rows=[](const Mat &m,Vec &E) -> pfloat {
        for(idxint col=0;col<m.cols();col++){
            for(idxint j=m.outerIndexPtr()[col];j<m.outerIndexPtr()[col+1];j++){
                idxint row=m.innerIndexPtr()[j];
                pfloat v=m.valuePtr()[j];
                E[row]=MAX(fabs(v),E[row]);
            }
        }
    };
    auto equil_cols=[](const Vec &E,Mat &m){
        for(idxint col=0;col<m.cols();col++){
            for(idxint j=m.outerIndexPtr()[col];j<m.outerIndexPtr()[col+1];j++){
                idxint row=m.innerIndexPtr()[j];
                m.valuePtr()[j]/=E[col];
            }
        }
    };
    auto equil_rows=[](const Vec &E,Mat &m){
        for(idxint col=0;col<m.cols();col++){
            for(idxint j=m.outerIndexPtr()[col];j<m.outerIndexPtr()[col+1];j++){
                idxint row=m.innerIndexPtr()[j];
                m.valuePtr()[j]/=E[row];
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
    for(idxint col=w->nm;col<w->n;col++){
        for(idxint j=w->G.outerIndexPtr()[col];j<w->G.outerIndexPtr()[col+1];j++){
            idxint row=w->G.innerIndexPtr()[j];
            w->G.valuePtr()[j]/=w->Gequil[row];
        }
    }
}

template<class PWork>
void KKTLDL2x2<PWork>::backscale(PWorkPtr w)
{
    idxint i;
    for( i=0; i < w->n; i++ ){ w->x[i] /= (w->xequil[i] * w->tau); }
    for( i=0; i < w->p; i++ ){ w->y[i] /= (w->tau); }
    for( i=0; i < w->m; i++ ){ w->z[i] /= (w->Gequil[i] * w->tau); }
    for( i=0; i < w->m; i++ ){ w->s[i] *= (w->Gequil[i] / w->tau); }
    for( i=0; i < w->n; i++ ){ w->c[i] *= w->xequil[i]; }
}

#include "la/la_eigen.h"
template kkt_ldl_3x3::spmat* get_spmat<LAEigen>(typename LAEigen::Mat &m);
template class KKTLDL2x2<PWorkLDL2x2<LAEigen> >;
/*
 * ECOS - Embedded Conic Solver.
 * Copyright (C) 2012-2015 A. Domahidi [domahidi@embotech.com],
 * Automatic Control Lab, ETH Zurich & embotech GmbH, Zurich, Switzerland.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


/* The KKT module.
 * Handles all computation related to KKT matrix:
 * - updating the matrix
 * - its factorization
 * - solving for search directions
 * - etc.
 */

#include "kkt.h"
#include "splamm.h"
#include "cone.h"
#include "acos.h"

#ifdef __cplusplus
extern "C"{
#endif
#include "ldl.h"
#ifdef __cplusplus
}
#endif

#include <math.h>
#include <iostream>

namespace ACOS{
namespace kkt_ldl_3x3{

/* Return codes */
#define KKT_PROBLEM (0)
#define KKT_OK      (1)

/* Factorization of KKT matrix. Just a wrapper for some LDL code */
idxint kkt_factor(kkt* KKT, pfloat eps, pfloat delta)
{
	idxint nd;

    /* returns n if successful, k if D (k,k) is zero */
	nd = LDL_numeric2(
				KKT->PKPt->n,	/* K and L are n-by-n, where n >= 0 */
				KKT->PKPt->jc,	/* input of size n+1, not modified */
				KKT->PKPt->ir,	/* input of size nz=Kjc[n], not modified */
				KKT->PKPt->pr,	/* input of size nz=Kjc[n], not modified */
				KKT->L->jc,		/* input of size n+1, not modified */
				KKT->Parent,	/* input of size n, not modified */
				KKT->Sign,      /* input, permuted sign vector for regularization */
                eps,            /* input, inverse permutation vector */
				delta,          /* size of dynamic regularization */
				KKT->Lnz,		/* output of size n, not defn. on input */
				KKT->L->ir,		/* output of size lnz=Lp[n], not defined on input */
				KKT->L->pr,		/* output of size lnz=Lp[n], not defined on input */
				KKT->D,			/* output of size n, not defined on input */
				KKT->work1,		/* workspace of size n, not defn. on input or output */
				KKT->Pattern,   /* workspace of size n, not defn. on input or output */
				KKT->Flag	    /* workspace of size n, not defn. on input or output */
    );
	return nd == KKT->PKPt->n ? KKT_OK : KKT_PROBLEM;
}


/**
 * Solves the permuted KKT system and returns the unpermuted search directions.
 *
 * On entry, the factorization of the permuted KKT matrix, PKPt,
 * is assumed to be up to date (call kkt_factor beforehand to achieve this).
 * The right hand side, Pb, is assumed to be already permuted.
 *
 * On exit, the resulting search directions are written into dx, dy and dz,
 * where these variables are permuted back to the original ordering.
 *
 * KKT->nitref iterative refinement steps are applied to solve the linear system.
 *
 * Returns the number of iterative refinement steps really taken.
 */
idxint kkt_solve(kkt* KKT, spmat* A, spmat* G, pfloat* Pb, pfloat* dx, pfloat* dy, pfloat* dz, idxint n, idxint p, idxint m, cone* C, idxint isinit, idxint nitref)
{

#define MTILDE (m+2*C->nsoc)

    idxint i, k, l, j, kk, kItRef;
	idxint dzoffset;
	idxint*  Pinv = KKT->Pinv;
	pfloat*    Px = KKT->work1;
	pfloat*   dPx = KKT->work2;
	pfloat*     e = KKT->work3;
    pfloat*    Pe = KKT->work4;
    pfloat* truez = KKT->work5;
    pfloat*   Gdx = KKT->work6;
    pfloat* ex = e;
    pfloat* ey = e + n;
    pfloat* ez = e + n+p;
    pfloat bnorm = 1.0 + norminf(Pb, n+p+MTILDE);
    pfloat nex = 0;
    pfloat ney = 0;
    pfloat nez = 0;
    pfloat nerr;
    pfloat nerr_prev = (pfloat)ACOS_NAN;
    pfloat error_threshold = bnorm*LINSYSACC;
    idxint nK = KKT->PKPt->n;

	/* forward - diagonal - backward solves: Px holds solution */
	LDL_lsolve2(nK, Pb, KKT->L->jc, KKT->L->ir, KKT->L->pr, Px );
	LDL_dsolve(nK, Px, KKT->D);
	LDL_ltsolve(nK, Px, KKT->L->jc, KKT->L->ir, KKT->L->pr);


	/* iterative refinement */
	for( kItRef=0; kItRef <= nitref; kItRef++ ){

        /* unpermute x & copy into arrays */
        unstretch(n, p, C, Pinv, Px, dx, dy, dz);

		/* compute error term */
        k=0; j=0;

		/* 1. error on dx*/
		/* ex = bx - A'*dy - G'*dz - DELTASTAT*dx */
        for( i=0; i<n; i++ ){ ex[i] = Pb[Pinv[k++]] - DELTASTAT*dx[i]; }

        if(A) sparseMtVm(A, dy, ex, 0, 0);
        sparseMtVm(G, dz, ex, 0, 0);
        nex = norminf(ex,n);

        /* error on dy */
        if( p > 0 ){
            for( i=0; i<p; i++ ){ ey[i] = Pb[Pinv[k++]] + DELTASTAT*dy[i]; }
            sparseMV(A, dx, ey, -1, 0);
            ney = norminf(ey,p);
        }


		/* --> 3. ez = bz - G*dx + V*dz_true */
        kk = 0; j=0;
		dzoffset=0;
        sparseMV(G, dx, Gdx, 1, 1);
        for( i=0; i<C->lpc->p; i++ ){
            ez[kk++] = Pb[Pinv[k++]] - Gdx[j++] + DELTASTAT*dz[dzoffset++];

        }
        for( l=0; l<C->nsoc; l++ ){
            for( i=0; i<C->soc[l].p; i++ ){
                ez[kk++] = i<(C->soc[l].p-1) ? Pb[Pinv[k++]] - Gdx[j++] + DELTASTAT*dz[dzoffset++] : Pb[Pinv[k++]] - Gdx[j++] - DELTASTAT*dz[dzoffset++];
            }
            ez[kk] = 0;
            ez[kk+1] = 0;
            k += 2;
            kk += 2;
        }

        for( i=0; i<MTILDE; i++) { truez[i] = Px[Pinv[n+p+i]]; }
        if( isinit == 0 ){
            scale2add(truez, ez, C);
        } else {
            vadd(MTILDE, truez, ez);
        }
        nez = norminf(ez,MTILDE);

        /* maximum error (infinity norm of e) */
        nerr = MAX( nex, nez);
        if( p > 0 ){ nerr = MAX( nerr, ney ); }
        // printf("kItRef:%d,nerr:%.17g\n",kItRef,nerr);

        /* CHECK WHETHER REFINEMENT BROUGHT DECREASE - if not undo and quit! */
        if( kItRef > 0 && nerr > nerr_prev ){
            /* undo refinement */
            for( i=0; i<nK; i++ ){ Px[i] -= dPx[i]; }
            kItRef--;
            break;
        }

        /* CHECK WHETHER TO REFINE AGAIN */
        if( kItRef == nitref || ( nerr < error_threshold ) || ( kItRef > 0 && nerr_prev < IRERRFACT*nerr ) ){
            break;
        }
        nerr_prev = nerr;

        /* permute */
        for( i=0; i<nK; i++) { Pe[Pinv[i]] = e[i]; }

        /* forward - diagonal - backward solves: dPx holds solution */
        LDL_lsolve2(nK, Pe, KKT->L->jc, KKT->L->ir, KKT->L->pr, dPx);
        LDL_dsolve(nK, dPx, KKT->D);
        LDL_ltsolve(nK, dPx, KKT->L->jc, KKT->L->ir, KKT->L->pr);

        /* add refinement to Px */
        for( i=0; i<nK; i++ ){ Px[i] += dPx[i]; }
	}

    // printf("nerr:%17.g\n",nerr);
	/* copy solution out into the different arrays, permutation included */
	unstretch(n, p, C, Pinv, Px, dx, dy, dz);

    return kItRef;
}


/**
 * Updates the permuted KKT matrix by copying in the new scalings.
 */
void kkt_update(spmat* PKP, idxint* P, cone *C)
{
	idxint i, j, k, conesize;
    pfloat eta_square, *q;
    pfloat d1, u0, u1, v1;
    idxint conesize_m1;


	/* LP cone */
    for( i=0; i < C->lpc->p; i++ ){ PKP->pr[P[C->lpc->kkt_idx[i]]] = -C->lpc->v[i] - DELTASTAT; }

	/* Second-order cone */
	for( i=0; i<C->nsoc; i++ ){
        getSOCDetails(&C->soc[i], &conesize, &eta_square, &d1, &u0, &u1, &v1, &q);
        conesize_m1 = conesize - 1;

        /* D */
        PKP->pr[P[C->soc[i].Didx[0]]] = -eta_square * d1 - DELTASTAT;
        for (k=1; k < conesize; k++) {
            PKP->pr[P[C->soc[i].Didx[k]]] = -eta_square - DELTASTAT;
        }

        /* v */
        j=1;
        for (k=0; k < conesize_m1; k++) {
            PKP->pr[P[C->soc[i].Didx[conesize_m1] + j++]] = -eta_square * v1 * q[k];
        }
        PKP->pr[P[C->soc[i].Didx[conesize_m1] + j++]] = -eta_square;

        /* u */
        PKP->pr[P[C->soc[i].Didx[conesize_m1] + j++]] = -eta_square * u0;
        for (k=0; k < conesize_m1; k++) {
            PKP->pr[P[C->soc[i].Didx[conesize_m1] + j++]] = -eta_square * u1 * q[k];
        }
        PKP->pr[P[C->soc[i].Didx[conesize_m1] + j++]] = +eta_square + DELTASTAT;
	}
}



/**
 * Initializes the (3,3) block of the KKT matrix to produce the matrix
 *
 * 		[0  A'  G']
 * K =  [A  0   0 ]
 *      [G  0  -I ]
 *
 * It is assumed that the A,G have been already copied in appropriately,
 * and that enough memory has been allocated (this is done in preproc.c module).
 *
 * Note that the function works on the permuted KKT matrix.
 */
void kkt_init(spmat* PKP, idxint* P, cone *C)
{
	idxint i, j, k, conesize;
    pfloat eta_square, *q;
    pfloat d1, u0, u1, v1;
    idxint conesize_m1;

	/* LP cone */
    for( i=0; i < C->lpc->p; i++ ){ PKP->pr[P[C->lpc->kkt_idx[i]]] = -1.0; }

	/* Second-order cone */
	for( i=0; i<C->nsoc; i++ ){
        getSOCDetails(&C->soc[i], &conesize, &eta_square, &d1, &u0, &u1, &v1, &q);
        conesize_m1 = conesize - 1;

        /* D */
        PKP->pr[P[C->soc[i].Didx[0]]] = -1.0;
        for (k=1; k < conesize; k++) {
            PKP->pr[P[C->soc[i].Didx[k]]] = -1.0;
        }

        /* v */
        j=1;
        for (k=0; k < conesize_m1; k++) {
            PKP->pr[P[C->soc[i].Didx[conesize_m1] + j++]] = 0.0;
        }
        PKP->pr[P[C->soc[i].Didx[conesize_m1] + j++]] = -1.0;

        /* u */
        PKP->pr[P[C->soc[i].Didx[conesize_m1] + j++]] = 0.0;
        for (k=0; k < conesize_m1; k++) {
            PKP->pr[P[C->soc[i].Didx[conesize_m1] + j++]] = 0.0;
        }
        PKP->pr[P[C->soc[i].Didx[conesize_m1] + j++]] = +1.0;
	}
}


/*
 * Builds KKT matrix.
 * We store and operate only on the upper triangular part.
 * Replace by or use in codegen.
 *
 * INPUT:      spmat* Gt - pointer to G'
 *             spmat* At - pointer to A'
 *               cone* C - pointer to cone struct
 *
 * OUTPUT:  idxint* Sign - pointer to vector of signs for regularization
 *              spmat* K - pointer to unpermuted upper triangular part of KKT matrix
 *         idxint* AttoK - vector of indices such that K[AtoK[i]] = A[i]
 *         idxint* GttoK - vector of indices such that K[GtoK[i]] = G[i]
 */
void createKKT_U(spmat* Gt, spmat* At, cone* C, idxint** S, spmat** K,
                 idxint* AttoK, idxint* GttoK)
{
    idxint i, j, k, l, r, row_stop, row, cone_strt, ks, conesize;
    idxint n = Gt->m;
    idxint m = Gt->n;
    idxint p = At ? At->n : 0;
    idxint nK, nnzK;
    pfloat *Kpr = NULL;
    idxint *Kjc = NULL, *Kir = NULL;
    idxint *Sign;
    /* Dimension of KKT matrix
     *   =   n (number of variables)
     *     + p (number of equality constraints)
     *     + m (number of inequality constraints)
     *     + 2*C->nsoc (expansion of SOC scalings)
     */
    nK = n + p + m;
    nK += 2*C->nsoc;

    /* Number of non-zeros in KKT matrix
     *   =   At->nnz (nnz of equality constraint matrix A)
     *     + Gt->nnz (nnz of inequality constraint matrix)
     *     + C->lpc.p (nnz of LP cone)
     *     + 3*[sum(C->soc[i].p)+1] (nnz of expanded soc scalings)
     *     + 6*C->nexc (3x3 hessian of the exponential cone)
     */
    nnzK = (At ? At->nnz : 0) + Gt->nnz + C->lpc->p;
    nnzK += n+p;
    for( i=0; i<C->nsoc; i++ ){
        nnzK += 3*C->soc[i].p+1;
    }

    /* Allocate memory for KKT matrix */
    Kpr = (pfloat *)MALLOC(nnzK*sizeof(pfloat));
    Kir = (idxint *)MALLOC(nnzK*sizeof(idxint));
    Kjc = (idxint *)MALLOC((nK+1)*sizeof(idxint));

    /* Allocate memory for sign vector */
    Sign = (idxint *)MALLOC(nK*sizeof(idxint));
    // std::cout<<"nnzK:"<<nnzK<<",nK:"<<nK<<std::endl;
    // std::cout<<"Sign:"<<Sign<<std::endl;

    /* Set signs for regularization of (1,1) block */
    for( ks=0; ks < n; ks++ ){
        Sign[ks] = +1; /* (1,1) block */
    }
    for( ks=n; ks < n+p; ks++){
        Sign[ks] = -1; /* (2,2) block */
    }
    for( ks=n+p; ks < n+p+C->lpc->p; ks++){
        Sign[ks] = -1; /* (3,3) block: LP cone */
    }
    ks = n+p+C->lpc->p;
    for( l=0; l<C->nsoc; l++){
        for (i=0; i<C->soc[l].p; i++) {
            Sign[ks++] = -1; /* (3,3) block: SOC, D */
        }
        Sign[ks++] = -1;     /* (3,3) block: SOC, v */
        Sign[ks++] = +1;     /* (3,3) block: SOC, u */
    }

    /* count the number of non-zero entries in K */
    k = 0;

    /* (1,1) block: the first n columns are empty */
    for (j=0; j<n; j++) {
        Kjc[j] = j;
        Kir[j] = j;
        Kpr[k++] = DELTASTAT;
    }

    /* Fill upper triangular part of K with values */
    /* (1,2) block: A' */
    i = 0; /* counter for non-zero entries in A or G, respectively */
    for( j=0; j<p; j++ ){
        /* A' */
        row = At->jc[j];
        row_stop = At->jc[j+1];
        if( row <= row_stop ){
            Kjc[n+j] = k;
            while( row++ < row_stop ){
                Kir[k] = At->ir[i];
                Kpr[k] = At->pr[i];
                AttoK[i++] = k++;
            }
        }
        Kir[k] = n+j;
        Kpr[k++] = -DELTASTAT;
    }
    /* (1,3) and (3,3) block: [G'; 0; -Vinit]
     * where
     *
     *   Vinit = blkdiag(I, blkdiag(I,1,-1), ...,  blkdiag(I,1,-1));
     *                        ^ #number of second-order cones ^
     *
     * Note that we have to prepare the (3,3) block accordingly
     * (put zeros for init but store indices that are used in KKT_update
     * of cone module)
     */

    /* LP cone */
    i = 0;
    for( j=0; j < C->lpc->p; j++ ){
        /* copy in G' */
        row = Gt->jc[j];
        row_stop = Gt->jc[j+1];
        if( row <= row_stop ){
            Kjc[n+p+j] = k;
            while( row++ < row_stop ){
                Kir[k] = Gt->ir[i];
                Kpr[k] = Gt->pr[i];
                GttoK[i++] = k++;
            }
        }
        /* -I for LP-cone */
        C->lpc->kkt_idx[j] = k;
        Kir[k] = n+p+j;
        Kpr[k] = -1.0;
        k++;
    }

    /* Second-order cones - copy in G' and set up the scaling matrix
     * which has the following structure:
     *
     *
     *    [ *             0  * ]
     *    [   *           *  * ]
     *    [     *         *  * ]       [ I   v  u  ]      I: identity of size conesize
     *  - [       *       *  * ]   =  -[ u'  1  0  ]      v: vector of size conesize - 1
     *    [         *     *  * ]       [ v'  0' -1 ]      u: vector of size conesize
     *    [           *   *  * ]
     *    [             * *  * ]
     *    [ 0 * * * * * * 1  0 ]
     *    [ * * * * * * * 0 -1 ]
     *
     * NOTE: only the upper triangular part (with the diagonal elements)
     *       is copied in here.
     */
    cone_strt = C->lpc->p;
    for( l=0; l < C->nsoc; l++ ){

        /* size of the cone */
        conesize = C->soc[l].p;

        /* go column-wise about it */
        for( j=0; j < conesize; j++ ){

           row = Gt->jc[cone_strt+j];
           row_stop = Gt->jc[cone_strt+j+1];
           if( row <= row_stop ){
               Kjc[n+p+cone_strt+2*l+j] = k;
               while( row++ < row_stop ){
                   Kir[k] = Gt->ir[i];
                   Kpr[k] = Gt->pr[i];
                   GttoK[i++] = k++;
               }
           }

           /* diagonal D */
           Kir[k] = n+p+cone_strt + 2*l + j;
           Kpr[k] = -1.0;
           C->soc[l].Didx[j] = k;
           k++;
        }

        /* v */
        Kjc[n+p+cone_strt+2*l+conesize] = k;
        for (r=1; r<conesize; r++) {
            Kir[k] = n+p+cone_strt + 2*l + r;
            Kpr[k] = 0;
            k++;
        }
        Kir[k] = n+p+cone_strt + 2*l + conesize;
        Kpr[k] = -1;
        k++;


        /* u */
        Kjc[n+p+cone_strt+2*l+conesize+1] = k;
        for (r=0; r<conesize; r++) {
            Kir[k] = n+p+cone_strt + 2*l + r;
            Kpr[k] = 0;
            k++;
        }
        Kir[k] = n+p+cone_strt + 2*l + conesize + 1;
        Kpr[k] = +1;
        k++;


        /* prepare index for next cone */
        cone_strt += C->soc[l].p;
    }

    /* return Sign vector and KKT matrix */
    *S = Sign;
    *K = ecoscreateSparseMatrix(nK, nK, nnzK, Kjc, Kir, Kpr);
}

/*
 * Line search according to Vandenberghe.
 */
pfloat lineSearch(pfloat* lambda, pfloat* ds, pfloat* dz, pfloat tau, pfloat dtau, pfloat kap, pfloat dkap, cone* C, kkt* KKT)
{
    idxint i, j, cone_start, conesize;
    pfloat rhomin, sigmamin, alpha, lknorm2, lknorm, lknorminv, rhonorm, sigmanorm, conic_step, temp;
    pfloat lkbar_times_dsk, lkbar_times_dzk, factor;
    pfloat* lk;
    pfloat* dsk;
    pfloat* dzk;
    pfloat* lkbar = KKT->work1;
    pfloat* rho = KKT->work2;
    pfloat* sigma = KKT->work2;
    pfloat minus_tau_by_dtau = -tau/dtau;
    pfloat minus_kap_by_dkap = -kap/dkap;


    /* LP cone */
    if( C->lpc->p > 0 ){
        rhomin = ds[0] / lambda[0];  sigmamin = dz[0] / lambda[0];
        for( i=1; i < C->lpc->p; i++ ){
            rho[0] = ds[i] / lambda[i];   if( rho[0] < rhomin ){ rhomin = rho[0]; }
            sigma[0] = dz[i] / lambda[i]; if( sigma[0] < sigmamin ){ sigmamin = sigma[0]; }
        }

        if( -sigmamin > -rhomin ){
            alpha = sigmamin < 0 ? 1.0 / (-sigmamin) : 1.0 / EPS;
        } else {
            alpha = rhomin < 0 ? 1.0 / (-rhomin) : 1.0 / EPS;
        }
    } else {
        alpha = 10;
    }

    /* tau and kappa */
    if( minus_tau_by_dtau > 0 && minus_tau_by_dtau < alpha )
    {
        alpha = minus_tau_by_dtau;
    }
    if( minus_kap_by_dkap > 0 && minus_kap_by_dkap < alpha )
    {
        alpha = minus_kap_by_dkap;
    }


    /* Second-order cone */
    cone_start = C->lpc->p;
    for( i=0; i < C->nsoc; i++ ){

        /* indices */
        conesize = C->soc[i].p;
        lk = lambda + cone_start;  dsk = ds + cone_start;  dzk = dz + cone_start;

        /* normalize */
        lknorm2 = lk[0]*lk[0] - eddot(conesize-1, lk+1, lk+1);
        if (lknorm2 <= 0.0)
            continue;

        lknorm = sqrt( lknorm2 );
        for( j=0; j < conesize; j++ ){ lkbar[j] = lk[j] / lknorm; }
        lknorminv = 1.0 / lknorm;

        /* calculate products */
        lkbar_times_dsk = lkbar[0]*dsk[0];
        for( j=1; j < conesize; j++ ){ lkbar_times_dsk -= lkbar[j]*dsk[j]; }
        lkbar_times_dzk = lkbar[0]*dzk[0];
        for( j=1; j < conesize; j++ ){ lkbar_times_dzk -= lkbar[j]*dzk[j]; }

        /* now construct rhok and sigmak, the first element is different */
        rho[0] = lknorminv * lkbar_times_dsk;
        factor = (lkbar_times_dsk+dsk[0])/(lkbar[0]+1);
        for( j=1; j < conesize; j++ ){ rho[j] = lknorminv*(dsk[j] - factor*lkbar[j]); }
        rhonorm = norm2(rho+1, conesize-1) - rho[0];

        sigma[0] = lknorminv * lkbar_times_dzk;
        factor = (lkbar_times_dzk+dzk[0])/(lkbar[0]+1);
        for( j=1; j < conesize; j++ ){ sigma[j] = lknorminv*(dzk[j] - factor*lkbar[j]); }
        sigmanorm = norm2(sigma+1, conesize-1) - sigma[0];

        /* update alpha */
        conic_step = 0;
        if( rhonorm > conic_step ){ conic_step = rhonorm; }
        if( sigmanorm > conic_step ){ conic_step = sigmanorm; }
        if( conic_step != 0 ){
            temp = 1.0 / conic_step;
            if( temp < alpha ){ alpha = temp; }
        }

        cone_start += C->soc[i].p;

    }

    /* saturate between STEPMIN and STEPMAX */
    if( alpha > STEPMAX ) alpha = STEPMAX;
    if( alpha < STEPMIN ) alpha = STEPMIN;

    /* return alpha */
    return alpha;
}

pfloat lineSearch(pfloat* lambda, pfloat* ds, pfloat* dz, pfloat tau, pfloat dtau, pfloat kap, pfloat dkap, cone* C, int nK)
{
    idxint i, j, cone_start, conesize;
    pfloat rhomin, sigmamin, alpha, lknorm2, lknorm, lknorminv, rhonorm, sigmanorm, conic_step, temp;
    pfloat lkbar_times_dsk, lkbar_times_dzk, factor;
    pfloat* lk;
    pfloat* dsk;
    pfloat* dzk;
    pfloat* lkbar = (pfloat *)MALLOC(nK*sizeof(pfloat));
    pfloat* rho = (pfloat *)MALLOC(nK*sizeof(pfloat));
    pfloat* sigma = rho;
    pfloat minus_tau_by_dtau = -tau/dtau;
    pfloat minus_kap_by_dkap = -kap/dkap;


    /* LP cone */
    if( C->lpc->p > 0 ){
        rhomin = ds[0] / lambda[0];  sigmamin = dz[0] / lambda[0];
        for( i=1; i < C->lpc->p; i++ ){
            rho[0] = ds[i] / lambda[i];   if( rho[0] < rhomin ){ rhomin = rho[0]; }
            sigma[0] = dz[i] / lambda[i]; if( sigma[0] < sigmamin ){ sigmamin = sigma[0]; }
        }

        if( -sigmamin > -rhomin ){
            alpha = sigmamin < 0 ? 1.0 / (-sigmamin) : 1.0 / EPS;
        } else {
            alpha = rhomin < 0 ? 1.0 / (-rhomin) : 1.0 / EPS;
        }
    } else {
        alpha = 10;
    }

    /* tau and kappa */
    if( minus_tau_by_dtau > 0 && minus_tau_by_dtau < alpha )
    {
        alpha = minus_tau_by_dtau;
    }
    if( minus_kap_by_dkap > 0 && minus_kap_by_dkap < alpha )
    {
        alpha = minus_kap_by_dkap;
    }


    /* Second-order cone */
    cone_start = C->lpc->p;
    for( i=0; i < C->nsoc; i++ ){

        /* indices */
        conesize = C->soc[i].p;
        lk = lambda + cone_start;  dsk = ds + cone_start;  dzk = dz + cone_start;

        /* normalize */
        lknorm2 = lk[0]*lk[0] - eddot(conesize-1, lk+1, lk+1);
        if (lknorm2 <= 0.0)
            continue;

        lknorm = sqrt( lknorm2 );
        for( j=0; j < conesize; j++ ){ lkbar[j] = lk[j] / lknorm; }
        lknorminv = 1.0 / lknorm;

        /* calculate products */
        lkbar_times_dsk = lkbar[0]*dsk[0];
        for( j=1; j < conesize; j++ ){ lkbar_times_dsk -= lkbar[j]*dsk[j]; }
        lkbar_times_dzk = lkbar[0]*dzk[0];
        for( j=1; j < conesize; j++ ){ lkbar_times_dzk -= lkbar[j]*dzk[j]; }

        /* now construct rhok and sigmak, the first element is different */
        rho[0] = lknorminv * lkbar_times_dsk;
        factor = (lkbar_times_dsk+dsk[0])/(lkbar[0]+1);
        for( j=1; j < conesize; j++ ){ rho[j] = lknorminv*(dsk[j] - factor*lkbar[j]); }
        rhonorm = norm2(rho+1, conesize-1) - rho[0];

        sigma[0] = lknorminv * lkbar_times_dzk;
        factor = (lkbar_times_dzk+dzk[0])/(lkbar[0]+1);
        for( j=1; j < conesize; j++ ){ sigma[j] = lknorminv*(dzk[j] - factor*lkbar[j]); }
        sigmanorm = norm2(sigma+1, conesize-1) - sigma[0];

        /* update alpha */
        conic_step = 0;
        if( rhonorm > conic_step ){ conic_step = rhonorm; }
        if( sigmanorm > conic_step ){ conic_step = sigmanorm; }
        if( conic_step != 0 ){
            temp = 1.0 / conic_step;
            if( temp < alpha ){ alpha = temp; }
        }

        cone_start += C->soc[i].p;

    }

    /* saturate between STEPMIN and STEPMAX */
    if( alpha > STEPMAX ) alpha = STEPMAX;
    if( alpha < STEPMIN ) alpha = STEPMIN;

    FREE(lkbar);
    FREE(rho);

    /* return alpha */
    return alpha;
}

}
}
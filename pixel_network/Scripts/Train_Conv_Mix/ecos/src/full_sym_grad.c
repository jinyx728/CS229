#include "ecos.h"
#include "ldl.h"
#include "full_sym_grad.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void full_sym_grad_update(spmat* PKP,idxint *P,cone *C,pfloat *s,pfloat *z)
{
	idxint i,j,k,r,t,l;
	idxint conesize,cone_strt_p;
	pfloat *st,*zt,*s1,*z1;
	pfloat s0,z0,z0s0;
	pfloat d0;
	pfloat factor;
	/* not sure what to do for linear inequality constraints */
	t=C->lpc->p;
	for(l=0;l<C->nsoc;l++){
		conesize=C->soc[l].p;
		st=s+t;s0=st[0];s1=st+1;
		zt=z+t;z0=zt[0];z1=zt+1;
		z0s0=z0*s0;
		d0=z0s0;
		for(j=0;j<conesize-1;j++){
			d0+=s1[j]*z1[j];
		}
		if(d0<0) d0=0;

		for(j=0;j<conesize;j++){
			if(j!=0&&k!=C->soc[l].Didx[j]){
				printf("Something is wrong with k, k:%d,Didx:%d\n",(int)k,(int)C->soc[l].Didx[j]);
			}
			k=C->soc[l].Didx[j];
			/* S */
			if(j==0){
				PKP->pr[P[k]]=s0;k++;
				for(r=0;r<conesize-1;r++){
					PKP->pr[P[k]]=s1[r];k++;
				}
			}
			else{
				PKP->pr[P[k]]=s1[j-1];k++;
				PKP->pr[P[k]]=s0;k++;
			}
			/* ZS D */
			if(j==0){
				PKP->pr[P[k]]=d0+DELTASTAT;k++;
			}
			else{
				PKP->pr[P[k]]=z0s0+DELTASTAT;k++;
			}
		}

		/* v */
		if(z0>=s0){
			factor=sqrt(s0)/sqrt(z0);
			for(j=1;j<conesize;j++){
				PKP->pr[P[k]]=factor*z1[j-1];k++;
			}
		}
		else{
			factor=sqrt(z0)/sqrt(s0);
			for(j=1;j<conesize;j++){
				PKP->pr[P[k]]=factor*s1[j-1];k++;
			}
		}
		PKP->pr[P[k]]=1+DELTASTAT;k++;

		t+=conesize;	
	}
}

void full_sym_grad_unstretch(idxint n, idxint p, cone *C, idxint *Pinv, pfloat *Px, pfloat *dx, pfloat *dy, pfloat *dz)
{
	// hack
	int i,j,k,l;
	memset(dx,0,n*sizeof(pfloat));
	memset(dy,0,p*sizeof(pfloat));
	j=0;k=n+p;
    for( l=0; l<C->nsoc; l++ ){
        for( i=0; i<C->soc[l].p; i++ ){ dz[j++] = Px[Pinv[k++]]; }
	}
}

void full_sym_grad_solve(kkt* KKT, spmat* A, spmat* G, pfloat* Pb, pfloat* dx, pfloat* dy, pfloat* dz, idxint n, idxint p, idxint m, cone* C, idxint isinit, idxint nitref)
{
    idxint i, k, l, j, kk, kItRef;
	idxint*  Pinv = KKT->Pinv;
	pfloat*    Px = KKT->work1;

    idxint nK = KKT->PKPt->n;
    LDL_lsolve2(nK, Pb, KKT->L->jc, KKT->L->ir, KKT->L->pr, Px );
    LDL_dsolve(nK, Px, KKT->D);
    LDL_ltsolve(nK, Px, KKT->L->jc, KKT->L->ir, KKT->L->pr);

    full_sym_grad_unstretch(n,p,C,Pinv,Px,dx,dy,dz);
}
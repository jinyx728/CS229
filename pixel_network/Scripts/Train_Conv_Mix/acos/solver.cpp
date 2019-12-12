//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "solver.h"
#include <string.h> 

using namespace ACOS;

/* Some internal defines */
#define ACOS_NOT_CONVERGED_YET (-87)  /* indicates no convergence yet    */

template<class SOLVER_POLICY>
idxint Solver<SOLVER_POLICY>::compareStatistics(StatsPtr infoA, StatsPtr infoB)
{
    if ( infoA->pinfres != ACOS_NAN && infoA->kapovert > 1){
        if( infoB->pinfres != ACOS_NAN) {
            /* A->pinfres != NAN, B->pinfres!=NAN */
            if ( ( infoA->gap > 0 && infoB->gap > 0 && infoA->gap < infoB->gap ) &&
                ( infoA->pinfres > 0 && infoA->pinfres < infoB->pres ) &&
                ( infoA->mu > 0 && infoA->mu < infoB->mu ) ){
                /* PRINTTEXT("BRANCH 1 "); */
                return 1;
            } else {
                /* PRINTTEXT("BRANCH 1 not OK"); */
                return 0;
            }
        } else {
            /* A->pinfres != NAN, B->pinfres==NAN */
            if ( ( infoA->gap > 0 && infoB->gap > 0 && infoA->gap < infoB->gap ) &&
               ( infoA->mu > 0 && infoA->mu < infoB->mu ) ){
                /* PRINTTEXT("BRANCH 2 "); */
                return 1;
            } else {
                /* PRINTTEXT("BRANCH 2 not OK"); */
                return 0;
            }
        }
    } else {
            /* A->pinfres == NAN or pinfres too large */
        if ( ( infoA->gap > 0 && infoB->gap > 0 && infoA->gap < infoB->gap ) &&
            ( infoA->pres > 0 && infoA->pres < infoB->pres ) &&
            ( infoA->dres > 0 && infoA->dres < infoB->dres ) &&
            ( infoA->kapovert > 0 && infoA->kapovert < infoB->kapovert) &&
            ( infoA->mu > 0 && infoA->mu < infoB->mu ) ){
            /* PRINTTEXT("BRANCH 3 OK "); */
            return 1;
        } else {
            /* PRINTTEXT("BRANCH 3 not OK"); */
            return 0;
        }
    }
}

template<class SOLVER_POLICY>
void Solver<SOLVER_POLICY>::saveIterateAsBest(PWorkPtr w)
{
    idxint i;
    LA::copy(w->x,w->best_x);
    LA::copy(w->y,w->best_y);
    LA::copy(w->z,w->best_z);
    LA::copy(w->s,w->best_s);
    w->best_kap = w->kap;
    w->best_tau = w->tau;
    w->best_cx = w->cx;
    w->best_by = w->by;
    w->best_hz = w->hz;
    w->best_info->pcost = w->info->pcost;
    w->best_info->dcost = w->info->dcost;
    w->best_info->pres = w->info->pres;
    w->best_info->dres = w->info->dres;
    w->best_info->pinfres = w->info->pinfres;
    w->best_info->dinfres = w->info->dinfres;
    w->best_info->gap = w->info->gap;
    w->best_info->relgap = w->info->relgap;
    w->best_info->mu = w->info->mu;
    w->best_info->kapovert = w->info->kapovert;
    w->best_info->iter = w->info->iter;
}

template<class SOLVER_POLICY>
void Solver<SOLVER_POLICY>::restoreBestIterate(PWorkPtr w)
{
    idxint i;
    LA::copy(w->best_x,w->x);
    LA::copy(w->best_y,w->y);
    LA::copy(w->best_z,w->z);
    LA::copy(w->best_s,w->s);
    w->kap = w->best_kap;
    w->tau = w->best_tau;
    w->cx = w->best_cx;
    w->by = w->best_by;
    w->hz = w->best_hz;
    w->info->pcost = w->best_info->pcost;
    w->info->dcost = w->best_info->dcost;
    w->info->pres = w->best_info->pres;
    w->info->dres = w->best_info->dres;
    w->info->pinfres = w->best_info->pinfres;
    w->info->dinfres = w->best_info->dinfres;
    w->info->gap = w->best_info->gap;
    w->info->relgap = w->best_info->relgap;
    w->info->mu = w->best_info->mu;
    w->info->kapovert = w->best_info->kapovert;
}


/*
 * This function is reponsible for checking the exit/convergence conditions of ACOS.
 * If one of the exit conditions is met, ACOS displays an exit message and returns
 * the corresponding exit code. The calling function must then make sure that ACOS
 * is indeed correctly exited, so a call to this function should always be followed
 * by a break statement.
 *
 *    If mode == 0, normal precisions are checked.
 *
 *    If mode != 0, reduced precisions are checked, and the exit display is augmented
 *                  by "Close to". The exitcodes returned are increased by the value
 *                  of mode.
 *
 * The primal and dual infeasibility flags w->info->pinf and w->info->dinf are raised
 * according to the outcome of the test.
 *
 * If none of the exit tests are met, the function returns ACOS_NOT_CONVERGED_YET.
 * This should not be an exitflag that is ever returned to the outside world.
 */
template<class SOLVER_POLICY>
idxint Solver<SOLVER_POLICY>::checkExitConditions(PWorkPtr w, idxint mode)
{
    pfloat feastol;
    pfloat abstol;
    pfloat reltol;

    /* set accuracy against which to check */
    if( mode == 0) {
        /* check convergence against normal precisions */
        feastol = w->stgs->feastol;
        abstol = w->stgs->abstol;
        reltol = w->stgs->reltol;
    } else {
        /* check convergence against reduced precisions */
        feastol = w->stgs->feastol_inacc;
        abstol = w->stgs->abstol_inacc;
        reltol = w->stgs->reltol_inacc;
    }

    /* Optimal? */
    if( ( -w->cx > 0 || -w->by - w->hz >= -abstol ) &&
        ( w->info->pres < feastol && w->info->dres < feastol ) &&
        ( w->info->gap < abstol || w->info->relgap < reltol  )){
        if( w->stgs->verbose ) {
            if( mode == 0) {
                PRINTTEXT("\nOPTIMAL (within feastol=%3.1e, reltol=%3.1e, abstol=%3.1e).", (double)MAX(w->info->dres, w->info->pres), (double)w->info->relgap, (double)w->info->gap);
            } else {
                PRINTTEXT("\nClose to OPTIMAL (within feastol=%3.1e, reltol=%3.1e, abstol=%3.1e).", (double)MAX(w->info->dres, w->info->pres), (double)w->info->relgap, (double)w->info->gap);
            }
        }
        w->info->pinf = 0;
        w->info->dinf = 0;
        return ACOS_OPTIMAL + mode;
    }

    /* Dual infeasible? */
    else if( (w->info->dinfres != ACOS_NAN) && (w->info->dinfres < feastol) && (w->tau < w->kap) ){
        if( w->stgs->verbose ) {
            if( mode == 0) {
                PRINTTEXT("\nUNBOUNDED (within feastol=%3.1e).", (double)w->info->dinfres );
            } else {
                PRINTTEXT("\nClose to UNBOUNDED (within feastol=%3.1e).", (double)w->info->dinfres );
            }
        }
        w->info->pinf = 0;
        w->info->dinf = 1;
        return ACOS_DINF + mode;
    }

    /* Primal infeasible? */
    else if( ((w->info->pinfres != ACOS_NAN && w->info->pinfres < feastol) && (w->tau < w->kap)) ||
            ( w->tau < w->stgs->feastol && w->kap < w->stgs->feastol && w->info->pinfres < w->stgs->feastol) ){
        if( w->stgs->verbose ) {
            if( mode == 0) {
                PRINTTEXT("\nPRIMAL INFEASIBLE (within feastol=%3.1e).", (double)w->info->pinfres );
            } else {
                PRINTTEXT("\nClose to PRIMAL INFEASIBLE (within feastol=%3.1e).", (double)w->info->pinfres );
            }
        }
        w->info->pinf = 1;
        w->info->dinf = 0;
        return ACOS_PINF + mode;
    }

    /* Indicate if none of the above criteria are met */
    else {
        return ACOS_NOT_CONVERGED_YET;
    }
}

/*
 * Initializes the solver.
 */
template<class SOLVER_POLICY>
idxint Solver<SOLVER_POLICY>::init(PWorkPtr w)
{
    idxint i, j, k, l, KKT_FACTOR_RETURN_CODE;
    pfloat rx, ry, rz;


    /* Initialize KKT matrix */
    KKT::init(w);  // #3

    KKT::init_rhs1(w,w->kkt->rhs1); // #1
    KKT::init_rhs2(w,w->kkt->rhs2); // #2

    /* get scalings of problem data */
    rx = LA::norm2(w->c); w->resx0 = MAX(1, rx);
    ry = LA::norm2(w->b); w->resy0 = MAX(1, ry);
    rz = LA::norm2(w->h); w->resz0 = MAX(1, rz);

    KKT_FACTOR_RETURN_CODE = KKT::factor(w); // #4

    /* check if factorization was successful, exit otherwise */
    if(  KKT_FACTOR_RETURN_CODE != SOLVER_POLICY::KKT::KKT_OK ){
    if( w->stgs->verbose ) PRINTTEXT("\nProblem in factoring KKT system, aborting.");
        return ACOS_FATAL;
    }


    /*
     * PRIMAL VARIABLES:
     *  - solve xhat = arg min ||Gx-h||_2^2  such that Ax = b
     *  - r = h - G*xhat
     * These two equations are solved by
     *
     * [ 0   A'  G' ] [ xhat ]     [ 0 ]
     * [ A   0   0  ] [  y   ]  =  [ b ]
     * [ G   0  -I  ] [ -r   ]     [ h ]
     *
     * and then take shat = r if alphap < 0, zbar + (1+alphap)*e otherwise
     * where alphap = inf{ alpha | sbar + alpha*e >= 0 }
     */

    /* Solve for RHS [0; b; h] */

    if(w->use_x0){
        LA::copy(w->x,w->kkt->dx1);
        w->kkt->dz1=w->h-w->G*w->x;
        KKT::bring2cone(w, w->kkt->dz1, w->s);  // #7
    }
    else{
        KKT::solve(w,w->kkt->rhs1,w->kkt->dx1,w->kkt->dy1,w->kkt->dz1,1/*is_init*/); // #5

        /* Copy out initial value of x */
        LA::copy(w->kkt->dx1,w->x); // #6

        /* Copy out -r into temporary variable */
        LA::copy(-w->kkt->dz1,w->kkt->work1);

        /* Bring variable to cone */
        KKT::bring2cone(w, w->kkt->work1, w->s);  // #7
    }

    /*
     * dual variables
     * solve (yhat,zbar) = arg min ||z||_2^2 such that G'*z + A'*y + c = 0
     *
     * we can solve this by
     *
     * [ 0   A'  G' ] [  x   ]     [ -c ]
     * [ A   0   0  ] [ yhat ]  =  [  0 ]
     * [ G   0  -I  ] [ zbar ]     [  0 ]
     *
     * and then take zhat = zbar if alphad < 0, zbar + (1+alphad)*e otherwise
     * where alphad = inf{ alpha | zbar + alpha*e >= 0 }
     */

    /* Solve for RHS [-c; 0; 0] */
    if(w->use_x0){
        LA::copy(w->kkt->dx1,w->kkt->dx2);
    }
    KKT::solve(w,w->kkt->rhs2,w->kkt->dx2,w->kkt->dy2,w->kkt->dz2,1/*is_init*/);

    /* Copy out initial value of y */
    LA::copy(w->kkt->dy2,w->y);

    /* Bring variable to cone */
    KKT::bring2cone(w, w->kkt->dz2, w->z );



    /* Prepare RHS1 - before this line RHS1 = [0; b; h], after it holds [-c; b; h] */
    KKT::update_rhs1(w,w->kkt->rhs1);

    /*
     * other variables
     */
    w->kap = 1.0;
    w->tau = 1.0;

    w->info->step = 0;
    w->info->step_aff = 0;
    w->info->dinf = 0;
    w->info->pinf = 0;

    return 0;
}



/*
 * Computes residuals.
 *
 * hrx = -A'*y - G'*z;  rx = hrx - c.*tau;  hresx = norm(rx,2);
 * hry = A*x;           ry = hry - b.*tau;  hresy = norm(ry,2);
 * hrz = s + G*x;       rz = hrz - h.*tau;  hresz = norm(rz,2);
 * rt = kappa + c'*x + b'*y + h'*z;
 */
template<class SOLVER_POLICY>
void Solver<SOLVER_POLICY>::computeResiduals(PWorkPtr w)
{
    /* rx = -A'*y - G'*z - c.*tau */
    w->rx=-w->Gt*w->z;
    if( w->p > 0 ) {
        w->rx-=w->At*w->y; 
    }
    w->hresx = LA::norm2(w->rx);
    w->rx-=w->c*w->tau;

    /* ry = A*x - b.*tau */

    if( w->p > 0 ){
        w->ry=w->A*w->x;
        w->hresy = LA::norm2(w->ry);
        w->ry-=w->b*w->tau;;
    } else {
        w->hresy = 0;
    }

    /* rz = s + G*x - h.*tau */
    w->rz=w->s+w->G*w->x;
    w->hresz = LA::norm2(w->rz);
    w->rz-=w->h*w->tau;

    /* rt = kappa + c'*x + b'*y + h'*z; */
    w->cx = LA::dot(w->c, w->x);
    w->by = w->p > 0 ? LA::dot(w->b, w->y) : 0.0;
    w->hz = LA::dot(w->h, w->z);
    w->rt = w->kap + w->cx + w->by + w->hz;

    /* Norms of x y z */
    w->nx = LA::norm2(w->x);
    w->ny = LA::norm2(w->y);
    w->ns = LA::norm2(w->s);
    w->nz = LA::norm2(w->z);
}



/*
 * Updates statistics.
 */
template<class SOLVER_POLICY>
void Solver<SOLVER_POLICY>::updateStatistics(PWorkPtr w)
{
    pfloat nry, nrz;

    StatsPtr info = w->info;

    /* mu = (s'*z + kap*tau) / (D+1) where s'*z is the duality gap */
    info->gap = LA::dot(w->s, w->z);
    info->mu = (info->gap + w->kap*w->tau) / (w->D + 1);

    info->kapovert = w->kap / w->tau;
    info->pcost = w->cx / w->tau;
    info->dcost = -(w->hz + w->by) / w->tau;

    /* relative duality gap */
    if( info->pcost < 0 ){ info->relgap = info->gap / (-info->pcost); }
    else if( info->dcost > 0 ){ info->relgap = info->gap / info->dcost; }
    else info->relgap = ACOS_NAN;

    /* residuals */
    nry = w->p > 0 ? LA::norm2(w->ry)/MAX(w->resy0+w->nx,1) : 0.0;
    nrz = LA::norm2(w->rz)/MAX(w->resz0+w->nx+w->ns,1);
    info->pres = MAX(nry, nrz) / w->tau;
    info->dres = LA::norm2(w->rx)/MAX(w->resx0+w->ny+w->nz,1) / w->tau;

    /* infeasibility measures
     *
     * CVXOPT uses the following:
     * info->pinfres = w->hz + w->by < 0 ? w->hresx / w->resx0 / (-w->hz - w->by) : NAN;
     * info->dinfres = w->cx < 0 ? MAX(w->hresy/w->resy0, w->hresz/w->resz0) / (-w->cx) : NAN;
     */
    info->pinfres = (w->hz + w->by)/MAX(w->ny+w->nz,1) < -w->stgs->reltol ? w->hresx / MAX(w->ny+w->nz,1) : ACOS_NAN;
    info->dinfres = w->cx/MAX(w->nx,1) < -w->stgs->reltol ? MAX(w->hresy/MAX(w->nx,1), w->hresz/MAX(w->nx+w->ns,1)) : ACOS_NAN;
}

template<class SOLVER_POLICY>
void Solver<SOLVER_POLICY>::printProgress(StatsPtr info)
{
    if( info->iter == 0 )
    {
        /* print header at very first iteration */
        PRINTTEXT("It     pcost       dcost      gap   pres   dres    k/t    mu     step   sigma     IR\n");
        PRINTTEXT("%2d  %+5.3e  %+5.3e  %+2.0e  %2.0e  %2.0e  %2.0e  %2.0e    ---    ---   %2d %2d  -\n",(int)info->iter, (double)info->pcost, (double)info->dcost, (double)info->gap, (double)info->pres, (double)info->dres, (double)info->kapovert, (double)info->mu, (int)info->nitref1, (int)info->nitref2);
    }  else {
         PRINTTEXT("%2d  %+5.3e  %+5.3e  %+2.0e  %2.0e  %2.0e  %2.0e  %2.0e  %6.4f  %2.0e  %2d %2d %2d\n",(int)info->iter, (double)info->pcost, (double)info->dcost, (double)info->gap, (double)info->pres, (double)info->dres, (double)info->kapovert, (double)info->mu, (double)info->step, (double)info->sigma,\
        (int)info->nitref1,\
        (int)info->nitref2,\
        (int)info->nitref3);
    }
}

template<class SOLVER_POLICY>
void Solver<SOLVER_POLICY>::deleteLastProgressLine( StatsPtr info )
{
    idxint i;
    idxint offset = 0;

    if( info->kapovert < 0 ) offset++;
    if( info->mu < 0) offset++;
    if( info->pres < 0 ) offset++;
    if (info->dres < 0 ) offset++;

    for (i=0; i<82+offset; i++) {
        PRINTTEXT("%c",8);
    }
}


template<class SOLVER_POLICY>
typename Solver<SOLVER_POLICY>::PWorkPtr Solver<SOLVER_POLICY>::setup(idxint n, idxint m, idxint p, idxint l, idxint ncones, idxint* q, idxint nex, pfloat* Gpr, idxint* Gjc, idxint* Gir, pfloat* Apr, idxint* Ajc, idxint* Air, pfloat* c, pfloat* h, pfloat* b,pfloat *x)
{
    PWorkPtr w=std::make_shared<PWork>();
    w->init(n,m,p,l,ncones,q,Gpr,Gjc,Gir,Apr,Ajc,Air,c,h,b,x);
    return w;
}

template<class SOLVER_POLICY>
void Solver<SOLVER_POLICY>::cleanup(PWorkPtr w)
{
    KKT::cleanup(w);
}


/*
 * Main solver routine.
 */
template<class SOLVER_POLICY>
idxint Solver<SOLVER_POLICY>::solve(PWorkPtr w)
{
    idxint i, initcode, KKT_FACTOR_RETURN_CODE;
    pfloat dtau_denom, dtauaff, dkapaff, sigma, dtau, dkap, bkap;
    idxint exitcode = ACOS_FATAL, interrupted = 0;
    pfloat pres_prev = (pfloat)ACOS_NAN;

    if(w->stgs->verbose){
        w->info->start_t=std::chrono::system_clock::now();
    }
    /* Initialize solver */
    initcode = init(w);
    if( initcode == ACOS_FATAL ){
        if( w->stgs->verbose ) PRINTTEXT("\nFatal error during initialization, aborting.");
        return ACOS_FATAL;
    }
    if(w->stgs->verbose){
        w->info->init_t=std::chrono::system_clock::now();
    }

    /* MAIN INTERIOR POINT LOOP ---------------------------------------------------------------------- */
    for( w->info->iter = 0; w->info->iter <= w->stgs->maxit ; w->info->iter++ ){
        computeResiduals(w);
        updateStatistics(w);

        /* Print info */
        if( w->stgs->verbose ) printProgress(w->info);

        /* SAFEGUARD: Backtrack to best previously seen iterate if
         *
         * - the update was bad such that the primal residual PRES has increased by a factor of SAFEGUARD, or
         * - the gap became negative
         *
         * If the safeguard is activated, the solver tests if reduced precision has been reached, and reports
         * accordingly. If not even reduced precision is reached, ECOS returns the flag ECOS_NUMERICS.
         */
        if( w->info->iter > 0 && (w->info->pres > SAFEGUARD*pres_prev || w->info->gap < 0) ){
            if( w->stgs->verbose ) deleteLastProgressLine( w->info );
            if( w->stgs->verbose ) PRINTTEXT("Unreliable search direction detected, recovering best iterate (%i) and stopping.\n", (int)w->best_info->iter);
            restoreBestIterate( w );

            /* Determine whether we have reached at least reduced accuracy */
            exitcode = checkExitConditions( w, ACOS_INACC_OFFSET );

            /* if not, exit anyways */
            if( exitcode == ACOS_NOT_CONVERGED_YET ){
                exitcode = ACOS_NUMERICS;
                if( w->stgs->verbose ) PRINTTEXT("\nNUMERICAL PROBLEMS (reached feastol=%3.1e, reltol=%3.1e, abstol=%3.1e).", (double)MAX(w->info->dres, w->info->pres), (double)w->info->relgap, (double)w->info->gap);
                break;
            } else {
                break;
            }
        }
        pres_prev = w->info->pres;


        /* Check termination criteria to full precision and exit if necessary */
        exitcode = checkExitConditions( w, 0 );
        if( exitcode == ACOS_NOT_CONVERGED_YET ){
            /*
             * Full precision has not been reached yet. Check for two more cases of exit:
             *  (i) min step size, in which case we assume we won't make progress any more, and
             * (ii) maximum number of iterations reached
             * If these two are not fulfilled, another iteration will be made.
             */

            /* Did the line search cock up? (zero step length) */
            if( w->info->iter > 0 && w->info->step == STEPMIN*GAMMA ){
                if( w->stgs->verbose ) deleteLastProgressLine( w->info );
                if( w->stgs->verbose ) PRINTTEXT("No further progress possible, recovering best iterate (%i) and stopping.", (int)w->best_info->iter );
                restoreBestIterate( w );

                /* Determine whether we have reached reduced precision */
                exitcode = checkExitConditions( w, ACOS_INACC_OFFSET );
                if( exitcode == ACOS_NOT_CONVERGED_YET ){
                    exitcode = ACOS_NUMERICS;
                    if( w->stgs->verbose ) PRINTTEXT("\nNUMERICAL PROBLEMS (reached feastol=%3.1e, reltol=%3.1e, abstol=%3.1e).", (double)MAX(w->info->dres, w->info->pres), (double)w->info->relgap, (double)w->info->gap);
                }
                break;
            }
            /* MAXIT reached? */
            else if( interrupted || w->info->iter == w->stgs->maxit ){
                const char *what = interrupted ? "SIGINT intercepted" : "Maximum number of iterations reached";
                /* Determine whether current iterate is better than what we had so far */
                if( compareStatistics( w->info, w->best_info) ){
                    if( w->stgs->verbose )
                        PRINTTEXT("%s, stopping.\n",what);
                } else
                {
                    if( w->stgs->verbose )
                        PRINTTEXT("%s, recovering best iterate (%i) and stopping.\n", what, (int)w->best_info->iter);
                    restoreBestIterate( w );
                }

                /* Determine whether we have reached reduced precision */
                exitcode = checkExitConditions( w, ACOS_INACC_OFFSET );
                if( exitcode == ACOS_NOT_CONVERGED_YET ){
                    exitcode = interrupted ? ACOS_SIGINT : ACOS_MAXIT;
                    if( w->stgs->verbose ) {
                        const char* what = interrupted ? "INTERRUPTED" : "RAN OUT OF ITERATIONS";
                        PRINTTEXT("\n%s (reached feastol=%3.1e, reltol=%3.1e, abstol=%3.1e).", what, (double)MAX(w->info->dres, w->info->pres), (double)w->info->relgap, (double)w->info->gap);
                    }
                }
                break;

            }
        } else {
            /* Full precision has been reached, stop solver */
            break;
        }



        /* SAFEGUARD:
         * Check whether current iterate is worth keeping as the best solution so far,
         * before doing another iteration
         */
        if (w->info->iter == 0) {
            /* we're at the first iterate, so there's nothing to compare yet */
            saveIterateAsBest( w );
        } else if( compareStatistics( w->info, w->best_info) ){
            /* PRINTTEXT("Better solution found, saving as best so far \n"); */
            saveIterateAsBest( w );
        }


        /* Compute scalings */

        if( KKT::updateScalings(w, w->s, w->z, w->lambda) == SOLVER_POLICY::KKT::OUTSIDE_CONE ){
            /* SAFEGUARD: we have to recover here */
            if( w->stgs->verbose ) deleteLastProgressLine( w->info );
            if( w->stgs->verbose ) PRINTTEXT("Slacks/multipliers leaving the cone, recovering best iterate (%i) and stopping.\n", (int)w->best_info->iter);
            restoreBestIterate( w );

            /* Determine whether we have reached at least reduced accuracy */
            exitcode = checkExitConditions( w, ACOS_INACC_OFFSET );
            if( exitcode == ACOS_NOT_CONVERGED_YET ){
                if( w->stgs->verbose ) PRINTTEXT("\nNUMERICAL PROBLEMS (reached feastol=%3.1e, reltol=%3.1e, abstol=%3.1e).", (double)MAX(w->info->dres, w->info->pres), (double)w->info->relgap, (double)w->info->gap);
                return ACOS_OUTCONE;

            } else {
                break;
            }
        }

        /* Update KKT matrix with scalings */
        KKT::update(w,w->C); // #7

        KKT_FACTOR_RETURN_CODE = KKT::factor(w);



        /* check if factorization was successful, exit otherwise */
        if(  KKT_FACTOR_RETURN_CODE != SOLVER_POLICY::KKT::KKT_OK ){
        if( w->stgs->verbose ) PRINTTEXT("\nProblem in factoring KKT system, aborting.");
            return ACOS_FATAL;
        }

        /* Solve for RHS1, which is used later also in combined direction */
        KKT::solve(w, w->kkt->rhs1, w->kkt->dx1, w->kkt->dy1, w->kkt->dz1,0/*is_init*/);

        /* AFFINE SEARCH DIRECTION (predictor, need dsaff and dzaff only) */
        KKT::RHS_affine(w,w->kkt->rhs2); // #8
        if(w->use_x0){
            LA::copy(w->kkt->dx1,w->kkt->dx2);
        }
        KKT::solve(w, w->kkt->rhs2, w->kkt->dx2, w->kkt->dy2, w->kkt->dz2,0/*is_init*/);

        /* dtau_denom = kap/tau - (c'*x1 + by1 + h'*z1); */
        dtau_denom = w->kap/w->tau - LA::dot(w->c, w->kkt->dx1) - LA::dot(w->b, w->kkt->dy1) - LA::dot(w->h, w->kkt->dz1);

        /* dtauaff = (dt + c'*x2 + by2 + h'*z2) / dtau_denom; */
        dtauaff = (w->rt - w->kap + LA::dot(w->c, w->kkt->dx2) + LA::dot(w->b, w->kkt->dy2) + LA::dot(w->h, w->kkt->dz2)) / dtau_denom;

        /* dzaff = dz2 + dtau_aff*dz1 */
        /* let dz2   = dzaff  we use this in the linesearch for unsymmetric cones*/
        /* and w_times_dzaff = Wdz_aff*/
        /* and dz2 = dz2+dtau_aff*dz1 will store the unscaled dz*/
        w->kkt->dz2+=dtauaff*w->kkt->dz1;
        KKT::scale(w, w->kkt->dz2, w->W_times_dzaff); // #8

        /* W\dsaff = -W*dzaff -lambda; */
        w->dsaff_by_W=-w->W_times_dzaff-w->lambda;

        /* dkapaff = -(bkap + kap*dtauaff)/tau; bkap = kap*tau*/
        dkapaff = -w->kap - w->kap/w->tau*dtauaff;

        /* Line search on W\dsaff and W*dzaff */
        w->info->step_aff = KKT::lineSearch(w,w->lambda, w->dsaff_by_W, w->W_times_dzaff, w->tau, dtauaff, w->kap, dkapaff); // #9


        /* Centering parameter */
        sigma = 1.0 - w->info->step_aff;
        sigma = sigma*sigma*sigma;
        if( sigma > SIGMAMAX ) sigma = SIGMAMAX;
        if( sigma < SIGMAMIN ) sigma = SIGMAMIN;
        w->info->sigma = sigma;

        /* COMBINED SEARCH DIRECTION */
        KKT::RHS_combined(w,w->kkt->rhs2); // #10
        KKT::solve(w,w->kkt->rhs2, w->kkt->dx2, w->kkt->dy2, w->kkt->dz2,0/*is_init*/);

        /* bkap = kap*tau + dkapaff*dtauaff - sigma*info.mu; */
        bkap = w->kap*w->tau + dkapaff*dtauaff - sigma*w->info->mu;

        /* dtau = ((1-sigma)*rt - bkap/tau + c'*x2 + by2 + h'*z2) / dtau_denom; */
        dtau = ((1-sigma)*w->rt - bkap/w->tau + LA::dot(w->c, w->kkt->dx2) + LA::dot(w->b, w->kkt->dy2) + LA::dot(w->h, w->kkt->dz2)) / dtau_denom;

        /* dx = x2 + dtau*x1;     dy = y2 + dtau*y1;       dz = z2 + dtau*z1; */
        w->kkt->dx2+=dtau*w->kkt->dx1;
        w->kkt->dy2+=dtau*w->kkt->dy1;
        w->kkt->dz2+=dtau*w->kkt->dz1;

        /*  ds_by_W = -(lambda \ bs + conelp_timesW(scaling,dz,dims)); */
        /* note that at this point w->dsaff_by_W holds already (lambda \ ds) */
        KKT::scale(w, w->kkt->dz2, w->W_times_dzaff); // #11
        w->dsaff_by_W=-(w->dsaff_by_W+w->W_times_dzaff);

        /* dkap = -(bkap + kap*dtau)/tau; */
        dkap = -(bkap + w->kap*dtau)/w->tau;

        /* Line search on combined direction */
        w->info->step = KKT::lineSearch(w, w->lambda, w->dsaff_by_W, w->W_times_dzaff, w->tau, dtau, w->kap, dkap) * w->stgs->gamma;

        /* Bring ds to the final unscaled form */
        /* ds = W*ds_by_W */
        KKT::scale(w, w->dsaff_by_W, w->dsaff);

        /* Update variables */
        w->x+=w->info->step*w->kkt->dx2;
        w->y+=w->info->step*w->kkt->dy2;
        w->z+=w->info->step*w->kkt->dz2;
        w->s+=w->info->step*w->dsaff;
        w->kap += w->info->step * dkap;
        w->tau += w->info->step * dtau;
    }

    /* scale variables back */
    KKT::backscale(w); // #12
    if(w->stgs->verbose){
        w->info->solve_t=std::chrono::system_clock::now();
        pfloat init_t=std::chrono::duration<pfloat>(w->info->init_t-w->info->start_t).count();
        pfloat solve_t=std::chrono::duration<pfloat>(w->info->solve_t-w->info->init_t).count();
        printf("\ninit:%f s,solve:%f s\n",init_t,solve_t);
    }
    return exitcode;
}

#include "policy_ldl_3x3.h"
template class Solver<PolicyLDL3x3>;
#include "policy_ldl_2x2.h"
template class Solver<PolicyLDL2x2>;
#include "policy_cg_eigen.h"
template class Solver<PolicyCGEigen>;

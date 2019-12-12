//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once

#include "glblopts.h"
#include "cone_data.h"
#include "kkt_data.h"
#include <memory>

namespace ACOS{
/* DEFAULT SOLVER PARAMETERS AND SETTINGS STRUCT ----------------------- */
#define MAXIT      (100)          /* maximum number of iterations         */
#define FEASTOL    (1E-8)        /* primal/dual infeasibility tolerance  */
#define ABSTOL     (1E-8)        /* absolute tolerance on duality gap    */
#define RELTOL     (1E-8)        /* relative tolerance on duality gap    */
#define FTOL_INACC (1E-4)        /* inaccurate solution feasibility tol. */
#define ATOL_INACC (5E-5)        /* inaccurate solution absolute tol.    */
#define RTOL_INACC (5E-5)        /* inaccurate solution relative tol.    */
#define GAMMA      (0.99)        /* scaling the final step length        */
#define STATICREG  (1)           /* static regularization: 0:off, 1:on   */
#define DELTASTAT  (7E-8)        /* regularization parameter             */
#define DELTA      (2E-7)        /* dyn. regularization parameter        */
#define EPS        (1E-13)  /* dyn. regularization threshold (do not 0!) */
#define VERBOSE    (1)           /* bool for verbosity; PRINTLEVEL < 3   */
#define NITREF     (50)          /* number of iterative refinement steps */
#define IRERRFACT  (6)           /* factor by which IR should reduce err */
#define LINSYSACC  (1E-14)       /* rel. accuracy of search direction    */
#define SIGMAMIN   (1E-4)        /* always do some centering             */
#define SIGMAMAX   (1.0)         /* never fully center                   */
#define STEPMIN    (1E-6)        /* smallest step that we do take        */
#define STEPMAX    (0.999)  /* largest step allowed, also in affine dir. */
#define SAFEGUARD  (500)         /* Maximum increase in PRES before
                                                ECOS_NUMERICS is thrown. */

/* EQUILIBRATION METHOD ------------------------------------------------ */
#define EQUILIBRATE (1)     /* use equlibration of data matrices? >0: yes */
#define EQUIL_ITERS (3)         /* number of equilibration iterations  */
#define RUIZ_EQUIL      /* define algorithm to use - if both are ... */
/*#define ALTERNATING_EQUIL*/ /* ... commented out no equlibration is used */


/* EXITCODES ----------------------------------------------------------- */
#define ACOS_OPTIMAL  (0)   /* Problem solved to optimality              */
#define ACOS_PINF     (1)   /* Found certificate of primal infeasibility */
#define ACOS_DINF     (2)   /* Found certificate of dual infeasibility   */
#define ACOS_INACC_OFFSET (10)  /* Offset exitflag at inaccurate results */
#define ACOS_MAXIT    (-1)  /* Maximum number of iterations reached      */
#define ACOS_NUMERICS (-2)  /* Search direction unreliable               */
#define ACOS_OUTCONE  (-3)  /* s or z got outside the cone, numerics?    */
#define ACOS_SIGINT   (-4)  /* solver interrupted by a signal/ctrl-c     */
#define ACOS_FATAL    (-7)  /* Unknown problem in solver                 */

/* SOME USEFUL MACROS -------------------------------------------------- */
#define MAX(X,Y)  ((X) < (Y) ? (Y) : (X))  /* maximum of 2 expressions   */
/* safe division x/y where y is assumed to be positive! */
#define SAFEDIV_POS(X,Y)  ( (Y) < EPS ? ((X)/EPS) : (X)/(Y) )
};

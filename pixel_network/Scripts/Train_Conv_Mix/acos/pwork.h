//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once

#include "acos.h"
#include <chrono>

namespace ACOS{
/* SETTINGS STRUCT ----------------------------------------------------- */
class Stgs{
public:
    pfloat gamma;                /* scaling the final step length        */
    pfloat delta;                /* regularization parameter             */
    pfloat eps;                  /* regularization threshold             */
    pfloat feastol;              /* primal/dual infeasibility tolerance  */
    pfloat abstol;               /* absolute tolerance on duality gap    */
    pfloat reltol;               /* relative tolerance on duality gap    */
    pfloat feastol_inacc; /* primal/dual infeasibility relaxed tolerance */
    pfloat abstol_inacc;  /* absolute relaxed tolerance on duality gap   */
    pfloat reltol_inacc;  /* relative relaxed tolerance on duality gap   */
    idxint nitref;               /* number of iterative refinement steps */
    idxint maxit;                /* maximum number of iterations         */
    idxint verbose;              /* verbosity bool for PRINTLEVEL < 3    */
};
typedef std::shared_ptr<Stgs> StgsPtr;


/* INFO STRUCT --------------------------------------------------------- */
class Stats{
public:
    pfloat pcost;
    pfloat dcost;
    pfloat pres;
    pfloat dres;
    pfloat pinf;
    pfloat dinf;
    pfloat pinfres;
    pfloat dinfres;
    pfloat gap;
    pfloat relgap;
    pfloat sigma;
    pfloat mu;
    pfloat step;
    pfloat step_aff;
    pfloat kapovert;
    idxint iter;
    idxint nitref1;
    idxint nitref2;
    idxint nitref3;

    // for timing
    std::chrono::system_clock::time_point start_t;
    std::chrono::system_clock::time_point init_t;
    std::chrono::system_clock::time_point solve_t;
};
typedef std::shared_ptr<Stats> StatsPtr;

/* ALL DATA NEEDED BY SOLVER ------------------------------------------- */
template<class LA>
class PWork{
public:
    typedef LA LA_type;
    typedef typename LA::Vec Vec;
    typedef typename LA::Mat Mat;
    typedef KKTData<LA> KKT;
    typedef std::shared_ptr<KKT> KKTPtr;
    typedef ConeData<LA> Cone;
    typedef std::shared_ptr<Cone> ConePtr;
    /* dimensions */
    idxint n;   /* number of primal variables x */
    idxint m;   /* number of conically constrained variables s */
    idxint p;   /* number of equality constraints */
    idxint D;   /* degree of the cone */
    idxint l;
    idxint ncones;

    /* variables */
    Vec x;  /* primal variables                    */
    Vec y;  /* multipliers for equality constaints */
    Vec z;  /* multipliers for conic inequalities  */
    Vec s;  /* slacks for conic inequalities       */
    Vec lambda; /* scaled variable                 */
    pfloat kap; /* kappa (homogeneous embedding)       */
    pfloat tau; /* tau (homogeneous embedding)         */

    /* best iterate seen so far */
    /* variables */
    Vec best_x;  /* primal variables                    */
    Vec best_y;  /* multipliers for equality constaints */
    Vec best_z;  /* multipliers for conic inequalities  */
    Vec best_s;  /* slacks for conic inequalities       */
    pfloat best_kap; /* kappa (homogeneous embedding)       */
    pfloat best_tau; /* tau (homogeneous embedding)         */
    pfloat best_cx;
    pfloat best_by;
    pfloat best_hz;
    StatsPtr best_info; /* info of best iterate               */

    /* temporary stuff holding search direction etc. */
    Vec dsaff;
    Vec dzaff;
    Vec W_times_dzaff;
    Vec dsaff_by_W;
    Vec saff;
    Vec zaff;

    /* cone */
    ConePtr C;

    /* problem data */
    Mat A,G,At,Gt;
    Vec c,b,h;

    /* equilibration vector */
    Vec xequil;
    Vec Aequil;
    Vec Gequil;

    /* scalings of problem data */
    pfloat resx0;  pfloat resy0;  pfloat resz0;

    /* residuals */
    Vec rx,ry,rz;   pfloat rt;
    pfloat hresx;  pfloat hresy;  pfloat hresz;

    /* norm iterates */
    pfloat nx,ny,nz,ns;

    /* temporary storage */
    pfloat cx;  pfloat by;  pfloat hz;  pfloat sz;

    /* KKT System */
    KKTPtr kkt;

    /* info struct */
    StatsPtr info;

    /* settings struct */
    StgsPtr stgs;

    bool use_x0;

    void init(idxint _n, idxint _m, idxint _p, idxint _l, idxint _ncones, idxint* q, pfloat* Gpr, idxint* Gjc, idxint* Gir, pfloat* Apr, idxint* Ajc, idxint* Air, pfloat* _c, pfloat* _h, pfloat* _b, pfloat *_x);
    void init_stgs();
};
};
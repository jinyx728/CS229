//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once

#include "acos.h"
#include "pwork.h"
namespace ACOS{
template<class SOLVER_POLICY>
class Solver{
public:
    typedef typename SOLVER_POLICY::LA LA;
    typedef typename LA::Vec Vec;
    typedef typename LA::Mat Mat;
    typedef typename SOLVER_POLICY::PWork PWork;
    typedef std::shared_ptr<PWork> PWorkPtr;
    typedef typename PWork::ConePtr ConePtr;
    typedef typename SOLVER_POLICY::KKT KKT;
    /* METHODS */

    /* set up work space
     * could be done by codegen
     *
     * Parameters:
     * idxint n        Number of variables
     * idxint m        Number of inequalities, number of rows of G
     * idxint p        Number of equality constraints
     * idxint l        Dimension of positive orthant
     * idxint ncones   Number of second order cones
     * idxint* q       Array of length 'ncones', defines the dimension of each cone
     * idxint nex      Number of exponential cones
     * pfloat* Gpr     Sparse G matrix data array (column compressed storage)
     * idxint* Gjc     Sparse G matrix column index array (column compressed storage)
     * idxint* Gir     Sparse G matrix row index array (column compressed storage)
     * pfloat* Apr     Sparse A matrix data array (column compressed storage) (can be all NULL if no equalities are present)
     * idxint* Ajc     Sparse A matrix column index array (column compressed storage) (can be all NULL if no equalities are present)
     * idxint* Air     Sparse A matrix row index array (column compressed storage) (can be all NULL if no equalities are present)
     * pfloat* c       Array of size n, cost function weights
     * pfloat* h       Array of size m, RHS vector of cone constraint
     * pfloat* b       Array of size p, RHS vector of equalities (can be NULL if no equalities are present)
     */
    static PWorkPtr setup(idxint n, idxint m, idxint p, idxint l, idxint ncones, idxint* q, idxint nex,
                       pfloat* Gpr, idxint* Gjc, idxint* Gir,
                       pfloat* Apr, idxint* Ajc, idxint* Air,
                       pfloat* c, pfloat* h, pfloat* b,pfloat *x);

    /* solve */
    static idxint solve(PWorkPtr w);

    /**
     * Cleanup: free memory (not used for embedded solvers, only standalone)
     *
     * Use the second argument to give the number of variables to NOT free.
     * This is useful if you want to use the result of the optimization without
     * copying over the arrays. One use case is the MEX interface, where we
     * do not want to free x,y,s,z (depending on the number of LHS).
     */
    static void cleanup(PWorkPtr w);

    static idxint compareStatistics(StatsPtr infoA, StatsPtr infoB);
    static void saveIterateAsBest(PWorkPtr w);
    static void restoreBestIterate(PWorkPtr w);
    static idxint checkExitConditions(PWorkPtr w, idxint mode);
    static idxint init(PWorkPtr w);
    static void computeResiduals(PWorkPtr w);
    static void updateStatistics(PWorkPtr w);
    static void printProgress(StatsPtr info);
    static void deleteLastProgressLine( StatsPtr info );
};
}
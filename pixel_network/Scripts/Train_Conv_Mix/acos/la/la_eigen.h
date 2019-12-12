//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#pragma once
#include "glblopts.h"
#include <Eigen/Dense>
#include <Eigen/SparseCore>

namespace ACOS{
class LAEigen{
public:
    typedef Eigen::Matrix<pfloat,Eigen::Dynamic,1> Vec;
    typedef Eigen::Matrix<idxint,Eigen::Dynamic,1> VecI;
    // typedef Eigen::SparseMatrix<pfloat,Eigen::ColMajor> Mat; // might changed to row major someday
    typedef Eigen::SparseMatrix<pfloat,Eigen::RowMajor> Mat;

    static Vec create_vec(const pfloat *d,idxint n);
    static Mat create_mat(idxint m,idxint n,const pfloat *pr,const idxint *jc,const idxint *ir);
    static Mat transpose(const Mat &m);
    static pfloat norm2(const Vec &v);
    static pfloat norminf(const Vec &v);
    static pfloat dot(const Vec &v1,const Vec &v2);
    static void copy(const Vec &src,Vec &tgt);
    static bool has_nan(const Vec &v);
    static bool has_inf(const Vec &v);
};
};
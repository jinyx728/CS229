//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "la_eigen.h"
#include <algorithm>
#include <vector>
#include <cmath>
using namespace ACOS;

LAEigen::Vec LAEigen::create_vec(const pfloat *d,idxint n)
{
    Vec v(n);std::copy(d,d+n,v.data());
    return v;
}

LAEigen::Mat LAEigen::create_mat(idxint m,idxint n,const pfloat *pr,const idxint *jc,const idxint *ir)
{
    Mat L;
    L.resize(m,n);
    typedef Eigen::Triplet<double> Triplet;
    std::vector<Triplet> triplets;
    for(int coli=0;coli<n;coli++){
        for(int pi=jc[coli];pi<jc[coli+1];pi++){
            pfloat v=pr[pi];
            int row=ir[pi];
            triplets.push_back(Triplet(row,coli,v));
        }
    }
    L.setFromTriplets(triplets.begin(),triplets.end());
    L.makeCompressed();
    return L;
}

LAEigen::Mat LAEigen::transpose(const Mat &m)
{
    return m.transpose();
}

pfloat LAEigen::norm2(const Vec &v)
{
    return v.norm();
}

pfloat LAEigen::norminf(const Vec &v)
{
    return v.lpNorm<Eigen::Infinity>();
}

pfloat LAEigen::dot(const Vec &v1,const Vec &v2)
{
    return v1.dot(v2);
}

void LAEigen::copy(const Vec &src,Vec &tgt)
{
    // tgt=src;
    std::copy(src.data(),src.data()+src.size(),tgt.data());
}

bool LAEigen::has_nan(const Vec &v)
{
    for(int i=0;i<v.size();i++){
        if(std::isnan(v[i]))
            return true;
    }
    return false;
}

bool LAEigen::has_inf(const Vec &v)
{
    for(int i=0;i<v.size();i++){
        if(std::isinf(v[i]))
            return true;
    }
    return false;
}
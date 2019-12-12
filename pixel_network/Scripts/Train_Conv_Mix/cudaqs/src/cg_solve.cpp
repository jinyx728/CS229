//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "cg_solve.h"

using namespace cudaqs;

int cudaqs::cg_solve(const SpringSystem *system, const SystemData &A, const Tensor<double> &x0, const Tensor<double> &b, Tensor<double> &x, CGData &cg_data,float tol, int max_iter)
{
    double rho_old = 0;
    double convergence_norm = 0;
    int iterations = 0;
    double FLT_MAX = 1e6;
    bool restart;
    double rho;
    double s_dot_q;
    double alpha;
    Tensor<double> &r=cg_data.r,&mr=cg_data.mr,&q=cg_data.q,&s=cg_data.s,&buffer=cg_data.buffer,&Ax=cg_data.Ax,&n=cg_data.n;

    x.copy(x0);
    // printf("cg_solve:x0:%f,b:%f\n",x0.norm(),b.norm());
    while(true){
        restart = (iterations == 0);
        if(restart){
            system->mul(A, x, Ax);
            //r=b-Ax;
            b.subtract(Ax, r);
            // printf("cg_solve:r:%f\n",r.norm());
        }
        convergence_norm = system->convergence_norm(A, r, n);
        if(convergence_norm <= tol){
            return iterations;
        }
        if(convergence_norm == max_iter){
            return -1;
        }
        iterations++;
        system->precondition(A, r, mr);
        rho = system->inner(mr, r);
        if(restart){
            s.copy(mr);
        }
        else{
            s*=rho/rho_old;
            s+=mr;
        }
        system->mul(A, s, q);
        // printf("cg_solve:s:%f,q:%f\n",s.norm(),q.norm());

        s_dot_q = system->inner(s, q);
        if(s_dot_q <= 0){
            printf("s_dot_q:%f,indefinite\n",s_dot_q);
            assert(false);
        }
        if(s_dot_q > 0){
            alpha = rho/s_dot_q;
        }
        else{
            //This never hits
            alpha = FLT_MAX;
        }
        //x += s*alpha;
        //r += q*(-alpha);
        s.multiply(alpha, buffer);
        x += buffer;
        q.multiply(-alpha, buffer);
        r += buffer;

        rho_old = rho;
    }
}

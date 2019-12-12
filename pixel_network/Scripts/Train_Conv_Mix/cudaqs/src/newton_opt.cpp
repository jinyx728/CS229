//#####################################################################
// Copyright 2019. Zhenglin Geng. 
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "newton_opt.h"
#include "cg_solve.h"
#include <cmath>

using namespace cudaqs;

void NewtonOpt::init(const SpringSystem *system_in, float newton_tol_in, float cg_tol_in, int cg_max_iter_in)
{
    this->system = system_in;
    this->newton_tol = newton_tol_in;
    this->cg_tol = cg_tol_in;
    this->cg_max_iter = cg_max_iter_in;
}

int NewtonOpt::solve(const Tensor<double> &anchor,const Tensor<double> &stiffen_anchor,OptData &data,Tensor<double> &x,int max_iter) const
{
    NewtonOptData &forward_data=data.forward_data;
    SystemData &system_data=data.system_data;
    CGData &cg_data=data.cg_data;
    Tensor<double> &dx=forward_data.dx,&J=forward_data.J,&neg_J=forward_data.neg_J,&x0=forward_data.x0,&Hdx=forward_data.Hdx;
    system_data.anchor.copy(anchor);
    system_data.stiffen_anchor.copy(stiffen_anchor);
    x.copy(anchor);
    dx.set_zero();

    double sqrt_N = std::sqrt(anchor.size[0]);
    int iters = 0;

    while(true){
        system->get_data(x,system_data);
        
        system->get_J(system_data,J);
        double norm_J=J.norm();
        if(norm_J<newton_tol){
            return SUCCESS;
        }
        system_data.J_rms=norm_J/sqrt_N;

        if(iters > 0){
            double Jdx = J.inner(dx);
            if(Jdx < 0){
                system->mul(system_data,dx,Hdx);
                double denom = dx.inner(Hdx);
                dx*=-Jdx/denom;
            }
            else{
                dx.set_zero();
            }
        }
        // printf("solve:J:%f,norm_J:%f\n",J.norm(),norm_J);
        J.negate(neg_J);
        int cg_iters=cg_solve(system, system_data, dx, neg_J, dx, cg_data, cg_tol, cg_max_iter);
        // printf("cg_iters:%d\n",cg_iters);

        x0.copy(x);
        double t=backtrack_line_search(x0,dx,data,norm_J,x);
        // printf("Line search,t:%f\n",t);

        if(t==0){
            // no further progress possible.
            if(norm_J<newton_tol*50)
                return INACCURATE_RESULT;
            else
                return SEARCH_DIR_FAIL;
        }
        iters++;
        if(iters >= max_iter){
            return MAX_ITER;
        }
    }
}

double NewtonOpt::backtrack_line_search(const Tensor<double>& x0, const Tensor<double>& dx,
OptData &data, double fx0, Tensor<double> &x, double t0, double alpha, double beta, int max_steps) const 
{
    double t = t0;
    double loss = fx0;
    int n_steps = 0;
    double min_f = fx0;
    double min_t = 0;
    NewtonOptData &forward_data=data.forward_data;
    SystemData &system_data=data.system_data;
    Tensor<double> &J=forward_data.J,&min_x=forward_data.min_x_buffer;

    auto f=[&](const Tensor<double> &x){
        system->get_data(x,system_data);
        system->get_J(system_data,J);
        return J.norm();
    };

    while(true){
        //x=x0+dx*t;
        dx.multiply(t, x);
        x.add(x0, x);
        loss = f(x);
        if(loss < min_f){
            min_f = loss;
            min_t = t;
            min_x.copy(x);
        }
        if(n_steps > max_steps){
            if(min_f<fx0){
                t=min_t;x.copy(min_x);
            }
            else{
                t=0;x.copy(x0);
            }
        }
        if(loss <= (1 - alpha*t)*fx0){
            break;
        }
        t *= beta;
        n_steps++;
    }
    return t;
}

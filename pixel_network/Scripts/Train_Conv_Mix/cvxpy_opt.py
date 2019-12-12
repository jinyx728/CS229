######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
import os
from os.path import join,isdir
import cvxpy as cp
from obj_io import Obj,read_obj,write_obj
from timeit import timeit
import numpy as np
class CvxpyOpt:
    def __init__(self,m,edges,l0):
        self.n_vts=len(m)
        self.edges=edges
        self.m=m
        self.l0=l0
        self.youngs_modulus=1
        self.use_spring_energy=True
        self.v=cp.Variable((self.n_vts,3))
        # self.v0=cp.Parameter((self.n_vts,3))
        self.verbose=True

    @timeit
    def solve(self,v0):
        # self.v0.value=v0
        self.constraints=self.form_edge_constraints(self.v,self.edges,self.l0*1.01)
        # self.constraints=[]
        self.prob=self.form_problem(self.m,self.v,v0,self.constraints)
        # self.prob.solve(warm_start=True,verbose=self.verbose,max_iters=50,feastol=1e-4,abstol=1e-4,reltol=1e-4,feastol_inacc=1e-3,abstol_inacc=5e-4,reltol_inacc=5e-4)
        self.prob.solve(warm_start=True,verbose=self.verbose)
        dual_values=[c.dual_value for c in self.constraints]
        return np.asarray(self.v.value),np.array(dual_values)

    @timeit
    def form_problem(self,m,v,v0,constraints):
        objective=self.form_objective(m,v,v0)
        if self.use_spring_energy:
            objective+=self.form_spring_energy(self.edges,self.l0,v,v0)
        # prob=cp.Problem(cp.Minimize(objective),constraints)
        prob=cp.Problem(cp.Minimize(0),constraints)
        return prob

    def form_objective(self,m,v,v0):
        obj=0
        for i in range(len(m)):
            obj+=m[i]*(v[i,:]-v0[i,:])**2
        return cp.sum(obj)
        # return cp.quad_form(v-v0,cp.diag(np.sqrt(m)))
        # cp.sum(m*(v-v0)**2)/self.n_vts

    def form_spring_energy(self,edges,L0,v,c):
        obj=0
        for edge_i in range(len(edges)):
            i0,i1=edges[edge_i]
            l0=L0[edge_i]
            c0,c1=c[i0],c[i1]
            lv=c1-c0
            l=np.linalg.norm(lv)
            lhat=lv/l
            k=self.youngs_modulus/l0
            v0,v1=v[i0],v[i1]
            obj+=k/2*(l-l0+cp.sum(lhat*((v1-c1)-(v0-c0))))**2
        return obj

    def form_edge_constraints(self,v,edges,rest_lengths):
        n_edges=len(edges)
        constraints=[]
        for i in range(n_edges):
            edge=edges[i]
            l=rest_lengths[i]
            v0=v[edge[0],:]
            v1=v[edge[1],:]
            constraints.append(cp.sum((v0-v1)**2)<=l**2)
        # print('# constraints:',len(constraints))
        return constraints
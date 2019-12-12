######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
import torch
import torch.nn as nn
import os
from os.path import join,isdir
import cvxpy as cp
from obj_io import Obj,read_obj,write_obj
from timeit import timeit
from cvxpy_opt import CvxpyOpt
from torch.autograd import Function
from cg_solve import cg_solve
import functools
import torch.multiprocessing as mp
from inequality_opt import InequalitySolver,InequalityOptSystem
import copy

def forward_solve(opt,vt):
    return opt.solve(vt)

def backward_solve(output_vt,output_grad,lmd,system,verbose=False):
    data=system.get_data(output_vt,lmd.view(-1,1))
    grad0=torch.zeros_like(output_vt,device=output_vt.device,dtype=output_vt.dtype)
    grad,cg_iters=cg_solve(system,data,grad0,output_grad,tol=1e-6,max_iterations=output_grad.numel())
    if verbose:
        print('cg_iters',cg_iters)
    if cg_iters==-1:
        print('cg_iters=-1,use output_grad')
        # return output_grad.unsqueeze(0)
    # print('cg_iters:',cg_iters)
    return grad.unsqueeze(0)

class CvxpyOptFunction(Function):

    @staticmethod
    def forward(ctx,input_vts,opts,system):
        n_samples=len(input_vts)
        device,dtype=input_vts.device,input_vts.dtype
        np_input_vts=input_vts.detach().cpu().numpy()
        np_output_info=[]
        for i in range(n_samples):
            np_output_info.append(forward_solve(opts[i],np_input_vts[i]))
        output_vts,lmd=zip(*np_output_info)
        output_vts=torch.tensor(list(output_vts)).to(device=device,dtype=dtype)
        lmd=torch.tensor(list(lmd)).to(device=device,dtype=dtype)
        ctx.save_for_backward(input_vts,output_vts,lmd)
        ctx.opts=opts
        ctx.system=system
        return output_vts

    @staticmethod
    def backward(ctx,output_grad):
        input_vts,output_vts,lmds=ctx.saved_tensors
        opts,system=ctx.opts,ctx.system
        n_samples=len(output_grad)
        grads=[None for i in range(n_samples)]

        for i in range(n_samples):
            # ddf_xc=-I
            grads[i]=backward_solve(output_vts[i],output_grad[i],lmds[i],system)
            if torch.any(torch.isnan(grads[i])):
                print('grads[{}]'.format(i),'has nan') 
                grads[i]=output_grad[i]

        return torch.cat(grads,dim=0),None,None,None

cvxpy_opt=CvxpyOptFunction.apply

class CvxpyOptModule(nn.Module):
    def __init__(self,opt,cg_system,batch_size=16):
        super(CvxpyOptModule,self).__init__()
        opts=[opt]
        for i in range(1,batch_size):
            opts.append(copy.deepcopy(opt))
        self.opts=opts
        self.system=cg_system

    def forward(self,x):
        return cvx_opt(x,self.opts,self.system)

def load_opt_data(res_ctx,ctx):
    max_num_constraints=ctx['max_num_constraints']
    tshirt_obj_path=join(res_ctx['shared_data_dir'],'flat_tshirt.obj')
    tshirt_obj=read_obj(tshirt_obj_path)
    n_vts=len(tshirt_obj.v)

    linear_edges=np.loadtxt(join(res_ctx['shared_data_dir'],'linear_edges.txt')).astype(int)
    linear_rest_lengths=np.loadtxt(join(res_ctx['shared_data_dir'],'mat_or_med_linear.txt'))
    edges=linear_edges
    l0=linear_rest_lengths
    if 'use_spring' in ctx and ctx['use_spring']:
        bend_edges=np.loadtxt(join(res_ctx['shared_data_dir'],'bend_edges.txt')).astype(int)
        bend_rest_lengths=np.loadtxt(join(res_ctx['shared_data_dir'],'mat_or_med_bend.txt'))
        edges=np.concatenate([linear_edges,bend_edges],axis=0)
        l0=np.concatenate([linear_rest_lengths,bend_rest_lengths])
        n_linear_edges=len(linear_edges)
        n_bend_edges=len(bend_edges)
        res_ctx['stiffness']=np.array([10/(1+np.sqrt(2))]*n_linear_edges+[2/(1+np.sqrt(2))]*n_bend_edges)
    if max_num_constraints>=0:
        edges=edges[:max_num_constraints]
        l0=l0[:max_num_constraints]

    # m=np.ones(n_vts)
    m=np.loadtxt(join(res_ctx['shared_data_dir'],'m.txt'))*1e4
    # l0*=1.01

    return m,edges,l0

def init_cvxpy_opt_module(res_ctx,ctx):
    device,dtype=ctx['device'],ctx['dtype']
    batch_size=ctx['batch_size']

    m,edges,l0=load_opt_data(res_ctx,ctx)
    n_vts=len(m)
    opt=CvxpyOpt(m=m,n_vts=n_vts,edges=edges,l0=l0)

    edges=torch.from_numpy(edges).to(device=device,dtype=torch.long)
    i0,i1=edges[:,0],edges[:,1]
    I0,I1=edges[:,:1],edges[:,1:2]
    l02=torch.from_numpy(l0*l0).to(device=device,dtype=dtype)
    m=torch.ones((n_vts,1),device=device,dtype=dtype)
    cg_system=InequalityOptSystem(m,l02,(i0,i1,I0,I1))
    return CvxpyOptModule(opt,cg_system,batch_size=batch_size)
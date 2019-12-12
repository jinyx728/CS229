######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import os
from os.path import join,isdir
import cvx_opt_cpp
from cvxpy_opt_func import load_opt_data
from patch_utils import PatchManager
from spring_opt import SpringOptSystem,NewtonOpt,spring_opt_backward
import time

def load_axial_data(res_ctx):
    shared_data_dir=res_ctx['shared_data_dir']
    i_path=join(shared_data_dir,'axial_spring_particles.txt')
    axial_i=np.loadtxt(i_path).astype(np.int)
    w_path=join(shared_data_dir,'axial_particle_weights.txt')
    axial_w=np.loadtxt(w_path).astype(np.float64)
    return axial_i,axial_w

def get_patch_axial_data(patch_vt_ids,axial_i,axial_w):
    patch_vt_set=set(patch_vt_ids)
    global_to_local_vt_map={patch_vt_ids[local_id]:local_id for local_id in range(len(patch_vt_ids))}
    patch_axial_i,patch_axial_w=[],[]
    for i in range(len(axial_i)):
        if all([vi in patch_vt_set for vi in axial_i[i]]):
            patch_axial_i.append([global_to_local_vt_map[vi] for vi in axial_i[i]])
            patch_axial_w.append(axial_w[i])
    return np.array(patch_axial_i).astype(np.int),np.array(patch_axial_w)


class SpringOptFunction(Function):

    @staticmethod
    def forward(ctx,opt,input_vts):
        output_vts=[]
        m_adjusted=[]
        success=[]
        batch_size=len(input_vts)
        for i in range(batch_size):
            x,data,s=opt.solve(input_vts[i])
            success.append(s)
            if not success:
                print(i,'not success,ignore layer')
                output_vts.append(input_vts[i].unsqueeze(0))
            else:
                output_vts.append(x.unsqueeze(0))
            m_adjusted.append(data['m_adjusted'].unsqueeze(0))
        output_vts=torch.cat(output_vts,dim=0)
        m_adjusted=torch.cat(m_adjusted,dim=0)
        ctx.save_for_backward(input_vts,output_vts,m_adjusted)
        ctx.opt=opt
        ctx.success=success
        return output_vts

    @staticmethod
    def backward(ctx,in_grad):
        input_vts,output_vts,m_adjusted=ctx.saved_tensors
        opt,success=ctx.opt,ctx.success
        system=opt.system
        batch_size=len(in_grad)
        out_grad=[]
        for i in range(batch_size):
            if not success:
                out_grad.append(in_grad[i].unsqueeze(0))
            else:
                # print('output_vts',output_vts.size())
                data=system.get_data(output_vts[i])
                data['c']=input_vts[i]
                data['m_adjusted']=m_adjusted[i]
                # print('c',data['c'].size(),'m_adjusted',data['m_adjusted'].size(),'vts',data['vts'])

                J=system.get_J(data)
                norm_J=torch.norm(J)
                data['J_rms']=norm_J/np.sqrt(len(output_vts[i]))
                g=spring_opt_backward(system,data,in_grad[i],cg_tol=1e-3,cg_max_iter=250)
                out_grad.append(g.unsqueeze(0))
        out_grad=torch.cat(out_grad,dim=0)
        return None,out_grad

spring_opt=SpringOptFunction.apply

class SpringOptModule(nn.Module):
    def __init__(self,res_ctx,ctx):
        super(SpringOptModule,self).__init__()
        self.device=ctx['device']
        self.dtype=ctx['dtype']
        self.patch_id=ctx['patch_id']

        m,edges,l0=load_opt_data(res_ctx,ctx)
        k=res_ctx['stiffness']
        harmonic_m=1/(1/m[edges[:,0]]+1/m[edges[:,1]])
        k*=harmonic_m

        if self.patch_id>=0:
            self.patch_manager=PatchManager(shared_data_dir=res_ctx['shared_data_dir'])
            self.patch_vt_ids=self.patch_manager.load_patch_vt_ids(self.patch_id)
            patch_edge_ids=self.patch_manager.get_patch_edge_ids(self.patch_vt_ids,edges)
            m=m[self.patch_vt_ids]
            l0=l0[patch_edge_ids]
            edges=self.patch_manager.get_patch_edges(self.patch_id,edges)
            k=k[patch_edge_ids]

        axial_i,axial_w=load_axial_data(res_ctx)
        if self.patch_id>=0:
            axial_i,axial_w=get_patch_axial_data(self.patch_vt_ids,axial_i,axial_w)
        m0,m1,m2,m3=m[axial_i[:,0]],m[axial_i[:,1]],m[axial_i[:,2]],m[axial_i[:,3]]
        axial_harmonic_m=4/(1/m0+1/m1+1/m2+1/m3)
        axial_k=axial_harmonic_m*1e-1 # magic number

        m*=2

        self.m=torch.from_numpy(m).to(dtype=self.dtype,device=self.device).view(-1,1)
        self.edges=torch.from_numpy(edges).to(dtype=torch.long,device=self.device).view(-1,2)
        self.l0=torch.from_numpy(l0).to(dtype=self.dtype,device=self.device).view(-1,1)
        self.k=torch.from_numpy(k).to(dtype=self.dtype,device=self.device).view(-1,1)
        axial_i=torch.from_numpy(axial_i).to(dtype=torch.long,device=self.device).view(-1,4)
        axial_w=torch.from_numpy(axial_w).to(dtype=self.dtype,device=self.device).view(-1,4)
        axial_k=torch.from_numpy(axial_k).to(dtype=self.dtype,device=self.device).view(-1,1)
        self.axial_data=(axial_i,axial_w,axial_k)

        self.system=SpringOptSystem(self.m,self.edges,self.l0,self.k,m_alpha=0.1,axial_data=self.axial_data) # magic number
        self.opt=NewtonOpt(self.system,newton_tol=1e-12,cg_tol=1e-3,cg_max_iter=max(len(self.m)//2,250)) # magic number

    def forward(self,x):
        return spring_opt(self.opt,x)







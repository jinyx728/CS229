######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import torch.nn as nn
from cg_solve import cg_solve
from topo_utils import get_linear_edges,get_bend_edges,filter_edges
import math

class MeshOptSystem:
    def __init__(self,rest_vts,fcs,front_vt_ids,back_vt_ids,bdry_ids,lambda0=0):
        self.bdry_ids=bdry_ids
        self.use_linear=True
        self.use_bend=True
        self.use_gravity=False
        n_vts,D=rest_vts.size()
        if self.use_linear:
            self.linear_edges=torch.from_numpy(get_linear_edges(fcs.cpu().numpy())).to(device=fcs.device,dtype=fcs.dtype)
            self.linear_I1=self.linear_edges[:,:1].repeat(1,D)
            self.linear_I2=self.linear_edges[:,1:2].repeat(1,D)
            self.linear_rest_lengths=self.get_lengths(rest_vts,self.linear_edges)
            self.linear_k=10/(1+math.sqrt(2))/self.linear_rest_lengths
            # print('linear',torch.min(self.linear_rest_lengths),torch.max(self.linear_rest_lengths))
            # print('avg,lengths',torch.mean(self.linear_rest_lengths),self.linear_rest_lengths.size())
        if self.use_bend:
            self.bend_edges=torch.from_numpy(filter_edges(get_bend_edges(fcs.cpu().numpy()),front_vt_ids,back_vt_ids)).to(device=fcs.device,dtype=fcs.dtype)
            self.bend_I1=self.bend_edges[:,:1].repeat(1,D)
            self.bend_I2=self.bend_edges[:,1:2].repeat(1,D)
            self.bend_rest_lengths=self.get_lengths(rest_vts,self.bend_edges)
            self.bend_k=2/(1+math.sqrt(2))/self.bend_rest_lengths
            # print('bend',torch.min(self.bend_rest_lengths),torch.max(self.bend_rest_lengths))

        density=5
        self.m=self.get_m(density,rest_vts,fcs)
        # print('mean m',torch.mean(self.m))

        if self.use_gravity:
            g=9.8
            self.mg=self.m*g

        self.lambda0=lambda0

    def get_lengths(self,vts,edges):
        return torch.norm(vts[edges[:,1]]-vts[edges[:,0]],dim=1,keepdim=True)

    def get_m(self,density,vts,fcs):
        I1,I2,I3=fcs[:,0],fcs[:,1],fcs[:,2]
        v1,v2,v3=vts[I1],vts[I2],vts[I3]
        st=torch.norm(torch.cross(v2-v1,v3-v1),dim=1)
        s=torch.zeros((vts.size(0)),dtype=vts.dtype,device=vts.device)
        s.scatter_add_(0,I1,st)
        s.scatter_add_(0,I2,st)
        s.scatter_add_(0,I3,st)
        s*=density/2/3
        return s.unsqueeze(1)

    def compute_data(self,vts,edges,eps=1e-8):
        d=vts[edges[:,1]]-vts[edges[:,0]]
        l=torch.norm(d,dim=1,keepdim=True)
        l=torch.clamp(l,min=eps)
        lhat=d/l
        return d,l,lhat

    def get_linear_data(self,vts):
        return self.compute_data(vts,self.linear_edges)
    
    def get_bend_data(self,vts):
        return self.compute_data(vts,self.bend_edges)

    def get_data(self,vts):
        data={}
        data['size']=vts.size()
        data['device']=vts.device
        data['dtype']=vts.dtype
        if self.use_linear:
            data['linear']=self.get_linear_data(vts)
        if self.use_bend:
            data['bend']=self.get_bend_data(vts)
        return data

    def add_J(self,data,J,k,rest_lengths,I1,I2):
        d,l,lhat=data
        je=k*(l-rest_lengths)*lhat
        J.scatter_add_(0,I1,-je)
        J.scatter_add_(0,I2,je)

    def add_linear_J(self,data,J):
        self.add_J(data,J,self.linear_k,self.linear_rest_lengths,self.linear_I1,self.linear_I2)

    def add_bend_J(self,data,J):
        self.add_J(data,J,self.bend_k,self.bend_rest_lengths,self.bend_I1,self.bend_I2)

    def get_J(self,data):
        J=torch.zeros(data['size'],device=data['device'],dtype=data['dtype'])
        if self.use_linear:
            self.add_linear_J(data['linear'],J)
        if self.use_bend:
            self.add_bend_J(data['bend'],J)
        if self.lambda0!=0:
            J+=self.m*(data['vt']-data['vt0'])*self.lambda0
        if self.use_gravity:
            J[:,:,1]+=-self.mg

        # J[self.bdry_ids]=0

        return J

    def add_Hu(self,data,u,Hu,k,rest_lengths,edges,I1,I2):
        d,l,lhat=data
        du=u[edges[:,1]]-u[edges[:,0]]
        d2u=(l-rest_lengths)*du+rest_lengths*(lhat*torch.sum(lhat*du,dim=1,keepdim=True))
        d2u*=k/l
        Hu.scatter_add_(0,I1,-d2u)
        Hu.scatter_add_(0,I2,d2u)

    def add_linear_Hu(self,data,u,Hu):
        self.add_Hu(data,u,Hu,self.linear_k,self.linear_rest_lengths,self.linear_edges,self.linear_I1,self.linear_I2)

    def add_bend_Hu(self,data,u,Hu):
        self.add_Hu(data,u,Hu,self.bend_k,self.bend_rest_lengths,self.bend_edges,self.bend_I1,self.bend_I2)

    def get_Hu(self,data,u):
        Hu=torch.zeros(data['size'],device=data['device'],dtype=data['dtype'])
        if self.use_linear:
            self.add_linear_Hu(data['linear'],u,Hu)
        if self.use_bend:
            self.add_bend_Hu(data['bend'],u,Hu)
        if self.lambda0!=0:
            Hu+=u*self.m*self.lambda0
        # Hu[bdry_ids]=0

        return Hu

    def mul(self,data,u):
        return self.get_Hu(data,self.get_Hu(data,u))

    def inner(self,x,y):
        return torch.sum(x*y)

    def convergence_norm(self,r):
        return torch.max(torch.norm(r,dim=1))

    def compute_egy(self,l,rest_lengths,k):
        return torch.sum(k*(l-rest_lengths)**2)/2

    def get_linear_egy(self,vt):
        _,l,_=self.get_linear_data(vt)
        return self.compute_egy(l,self.linear_rest_lengths,self.linear_k)

    def get_bend_egy(self,vt):
        _,l,_=self.get_bend_data(vt)
        return self.compute_egy(l,self.bend_rest_lengths,self.bend_k)

    def get_total_egy(self,vt):
        egy=0
        if self.use_linear:
            egy+=self.get_linear_egy(vt)
        if self.use_bend:
            egy+=self.get_bend_egy(vt)
        if self.use_gravity:
            egy+=-torch.sum(self.mg*vt[:,1])
        return egy


class NewtonStepper:
    def __init__(self,rest_vts,fcs,front_vt_ids,back_vt_ids,bdry_ids,lambda0=0,cg_tol=1e-3,cg_max_iter=1000,n_steps=1):
        self.system=MeshOptSystem(rest_vts,fcs,front_vt_ids,back_vt_ids,bdry_ids,lambda0=lambda0)
        self.cg_tol=cg_tol
        self.cg_max_iter=cg_max_iter

    def step(self,vts,vt0=None):
        system=self.system
        data=system.get_data(vts)
        data['vt']=vts
        data['vt0']=vt0
        # print('pd',torch.mean(data['linear'][1]),data['linear'][1].size())
        J=system.get_J(data)
        HJ=system.get_Hu(data,J)
        dx0=torch.zeros_like(vts,dtype=vts.dtype,device=vts.device)
        dx,cg_iters=cg_solve(system,data,dx0,HJ,tol=self.cg_tol,max_iterations=self.cg_max_iter)
        return (vts-dx),cg_iters if cg_iters>0 else self.cg_max_iter



######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import torch.nn as nn
import numpy as np
from cg_solve import cg_solve
import os
from os.path import join,isdir,isfile

def backtrack_line_search(x0,dx,df,f,fx0,t0=1,alpha=0.01,beta=0.5,max_steps=8):
    dfdx=torch.sum(df*dx)
    t=t0
    loss=fx0
    n_steps=0
    min_f=fx0
    min_t=0
    min_x=x0
    while True: 
        x=x0+t*dx
        loss=f(x)

        if loss<min_f:
            min_f=loss
            min_t=t
            min_x=x
        if n_steps>max_steps:
            if min_f<fx0:
                f=min_f
                t=min_t
                x=min_x
            else:
                f=fx0
                t=0
                x=x0
            break

        # if loss<=fx0+alpha*t*dfdx:
        if loss<=(1-alpha*t)*fx0:
            break
        t*=beta
        n_steps+=1

    return t,x

def m_ratio_line_search(x0,dx,system,m_ratio_bound,beta=0.5,max_steps=8):
    t=1
    while True:
        x=x0+dx*t
        data=system.get_data(x)
        m_adjusted,_=system.get_m_adjusted(data,system.m_alpha)
        m_ratio=torch.max(m_adjusted)/torch.min(m_adjusted)
        if m_ratio<m_ratio_bound:
            return t
        t*=beta
        n_steps+=1
        if n_steps>max_steps:
            return 0


class SpringOptSystem:
    def __init__(self,stiffen_anchors_net,stiffen_anchors_reg,edges,l0,k,m_alpha=0.1,g=0,axial_data=None):
        self.stiffen_anchors_net=stiffen_anchors_net
        self.stiffen_anchors_reg=stiffen_anchors_reg
        self.edges=edges
        self.l0=l0
        self.k=k
        self.m_alpha=m_alpha
        D=3
        self.i0,self.i1=self.edges[:,:1],self.edges[:,1:]
        self.I0=self.i0.repeat(1,D)
        self.I1=self.i1.repeat(1,D)
        self.g=g
        self.use_axial_springs=False
        if axial_data is not None:
            self.use_axial_springs=True
            self.axial_i,self.axial_w,self.axial_k=axial_data
        # print('i0',self.i0.size(),'i1',self.i1.size(),'I0',self.I0.size(),'I1',self.I1.size())

    def get_data(self,vts):
        data={}
        data['size']=vts.size()
        data['device']=vts.device
        data['dtype']=vts.dtype
        data['vts']=vts

        d=vts[self.edges[:,1]]-vts[self.edges[:,0]]
        l=torch.norm(d,dim=1,keepdim=True)
        clamp_map=(l<1e-10).view(-1)
        l[clamp_map]=1
        if torch.sum(clamp_map)>0:
            print('# l=0:',torch.sum(clamp_map))
        lhat=d/l # safe divide
        l0_over_l=self.l0/l

        d[clamp_map]=0
        l[clamp_map]=0
        lhat[clamp_map]=0
        l0_over_l[clamp_map]=0

        data['edge_data']=(d,l,lhat)
        data['l0_over_l']=l0_over_l
        data['edge_clamp_map']=clamp_map
        # data['m_adjusted'],data['D']=self.get_m_adjusted(data,self.m_alpha)
        return data

    def get_m_adjusted(self,data,alpha):
        assert(False)
        m=self.m
        k=self.k
        l0=self.l0
        d,l,lhat=data['edge_data']
        l0_over_l=data['l0_over_l']
        r=k*(1-l0_over_l)
        r[data['edge_clamp_map']]=0

        pos_map=r>0
        pos_r=r[pos_map].view(-1,1)
        pos_i0=self.i0[pos_map].view(-1,1)
        pos_i1=self.i1[pos_map].view(-1,1)
        neg_map=r<0
        neg_r=r[neg_map].view(-1,1)
        neg_i0=self.i0[neg_map].view(-1,1)
        neg_i1=self.i1[neg_map].view(-1,1)

        neg_kI=torch.zeros_like(m,dtype=m.dtype,device=m.device)
        neg_kI.scatter_add_(0,neg_i0,neg_r)
        neg_kI.scatter_add_(0,neg_i1,neg_r)
        D=neg_kI+m

        bound=alpha*m-neg_kI
        clamp_map=D<bound
        D[clamp_map]=bound[clamp_map]
        m_adjusted=D-neg_kI

        if torch.sum(clamp_map)>0:
            # print('# D clamped',torch.sum(clamp_map).item())
            pass

        pos_kI=torch.zeros_like(m,dtype=m.dtype,device=m.device)
        pos_kI.scatter_add_(0,pos_i0,pos_r)
        pos_kI.scatter_add_(0,pos_i1,pos_r)

        # print('avg m_adjusted:',torch.mean(m_adjusted).item())

        return m_adjusted,pos_kI

    def get_J(self,data):
        J=data['stiffen_anchors_net']*(data['vts']-data['anchors_net'])+data['stiffen_anchors_reg']*(data['vts']-data['anchors_reg'])
        d,l,lhat=data['edge_data']
        r=self.k*(l-self.l0)*lhat
        J.scatter_add_(0,self.I0,-r)
        J.scatter_add_(0,self.I1,r)

        if self.g!=0:
            J[:,1]+=self.m[:,0]*self.g*data['vts'][:,1]
        if self.use_axial_springs:
            self.add_axial_J(data,J)

        return J

    def get_Hu(self,data,u):
        d,l,lhat=data['edge_data']
        l0_over_l=data['l0_over_l']
        def dot(a,b):
            return torch.sum(a*b,dim=1,keepdim=True)

        du=u[self.edges[:,1]]-u[self.edges[:,0]]
        if self.use_m_adjusted:
            r=self.k*((1-l0_over_l)*du+l0_over_l*dot(lhat,du)*lhat)
        else:
            lu=dot(lhat,du)*lhat
            lv=du-lu
            c=torch.clamp(1-l0_over_l,0)
            r=self.k*(c*lv+lu)
        Hu=(data['stiffen_anchors_net']+data['stiffen_anchors_reg'])*u
        Hu.scatter_add_(0,self.I0,-r)
        Hu.scatter_add_(0,self.I1,r)

        if self.use_axial_springs:
            self.add_axial_Hu(data,u,Hu)

        return Hu

    def precondition(self,data,x):
        return x
        # return self.jacobi_precondition(data,x)
        # return self.block_jacobi_precondition(data,x)

    def jacobi_precondition(self,data,x):
        assert(False)
        _,l,lhat=data['edge_data']
        l0_over_l=data['l0_over_l']
        r=self.k*l0_over_l*(lhat**2)
        D=data['D'].repeat(1,x.size(1))
        D.scatter_add_(0,self.I0,r)
        D.scatter_add_(0,self.I1,r)
        return x/D

    def block_jacobi_precondition(self,data,x):
        assert(False)
        _,l,lhat=data['edge_data']
        l0_over_l=data['l0_over_l']
        n_edges=len(lhat)
        n_vts,D=x.size()
        LL=torch.bmm(lhat.view(n_edges,D,1),lhat.view(n_edges,1,D)).view(n_edges,-1)
        LL*=self.k*l0_over_l
        M=torch.zeros((n_vts,D*D),device=x.device,dtype=x.dtype)
        M.scatter_add_(0,self.i0.repeat(1,D*D),LL)
        M.scatter_add_(0,self.i1.repeat(1,D*D),LL)
        M[:,[0,4,8]]+=data['D']
        M=M.view(n_vts,D,D)
        M=torch.inverse(M)
        x=torch.bmm(M,x.view(n_vts,D,1))
        return x.view(n_vts,D)

    def mul(self,data,u):
        return self.get_Hu(data,u)

    def inner(self,x,y):
        return torch.sum(x*y)

    def convergence_norm(self,data,r):
        return torch.max(torch.norm(r,dim=1))/data['J_rms']
        # return torch.norm(r)/data['rs']

    def compute_energy(self,data):
        return self.compute_edge_energy(data)+self.compute_anchor_energy(data)

    def compute_edge_energy(self,data):
        _,l,_=data['edge_data']
        energy=0.5*torch.sum(self.k*(self.l0-l)**2)
        if self.use_axial_springs:
            energy+=self.compute_axial_energy(data)
        return energy

    def compute_axial_energy(self,data):
        x=data['vts']
        r=self.axial_gather(x,self.axial_i,self.axial_w)
        return 0.5*torch.sum(self.axial_k*r**2)


    def compute_anchors_net_energy(self,data):
        return 0.5*torch.sum(data['stiffen_anchors_net']*(data['vts']-data['anchors_net'])**2)

    def axial_gather(self,x,i,w):
        i0,i1,i2,i3=i[:,0],i[:,1],i[:,2],i[:,3]
        w0,w1,w2,w3=w[:,:1],w[:,1:2],w[:,2:3],w[:,3:4]
        # print('i0',i0.size(),'x[i0]',x[i0].size(),'w',w0.size())
        # print('i1',i1.size(),'x[i1]',x[i1].size(),'w',w1.size())
        # print('i2',i2.size(),'x[i2]',x[i2].size(),'w',w2.size())
        # print('i3',i3.size(),'x[i3]',x[i3].size(),'w',w3.size())
        r=x[i0]*w0+x[i1]*w1-x[i2]*w2-x[i3]*w3
        return r

    def axial_scatter(self,x,r,i,w):
        D=x.size(1)
        I0,I1,I2,I3=i[:,:1].repeat(1,D),i[:,1:2].repeat(1,D),i[:,2:3].repeat(1,D),i[:,3:4].repeat(1,D)
        w0,w1,w2,w3=w[:,:1],w[:,1:2],w[:,2:3],w[:,3:4]
        x.scatter_add_(0,I0,r*w0)
        x.scatter_add_(0,I1,r*w1)
        x.scatter_add_(0,I2,r*(-w2))
        x.scatter_add_(0,I3,r*(-w3))

    def add_axial_J(self,data,J):
        x=data['vts']
        r=self.axial_gather(x,self.axial_i,self.axial_w)
        r*=self.axial_k
        self.axial_scatter(J,r,self.axial_i,self.axial_w)

    def add_axial_Hu(self,data,u,Hu):
        r=self.axial_gather(u,self.axial_i,self.axial_w)
        r*=self.axial_k
        self.axial_scatter(Hu,r,self.axial_i,self.axial_w)


class NewtonOpt:
    def __init__(self,system,newton_tol=1e-3,cg_tol=1e-3,cg_max_iter=1000):
        self.system=system
        self.newton_tol=newton_tol
        self.cg_tol=cg_tol
        self.cg_max_iter=cg_max_iter
        self.use_m_adjusted=False
        self.system.use_m_adjusted=self.use_m_adjusted

    def solve(self,anchors_net,anchors_reg,max_iter=200):
        system=self.system
        x=anchors_reg
        dx=torch.zeros_like(anchors_reg,dtype=anchors_reg.dtype,device=anchors_reg.device)
        sqrt_N=np.sqrt(len(x))
        iters=0
        def dot(a,b):
            return torch.sum(a*b)
        success=False
        while True:
            data=system.get_data(x)
            if self.use_m_adjusted:
                assert(False)
                m_adjusted_tmp,pos_kI=system.get_m_adjusted(data,system.m_alpha)
                if iters>0:
                    torch.max(m_adjusted,m_adjusted_tmp,out=m_adjusted)
                else:
                    m_adjusted=m_adjusted_tmp

                data['m_adjusted']=m_adjusted
                data['D']=m_adjusted+pos_kI
            else:
                stiffen_anchors_reg=system.stiffen_anchors_reg
                stiffen_anchors_net=system.stiffen_anchors_net

            data['anchors_net']=anchors_net
            data['anchors_reg']=anchors_reg
            data['stiffen_anchors_net']=stiffen_anchors_net
            data['stiffen_anchors_reg']=stiffen_anchors_reg

            # print('D,min:',torch.min(D).item(),'max:',torch.max(D).item())

            J=system.get_J(data)
            norm_J=torch.norm(J)
            if norm_J<self.newton_tol:
                # print('converged,iter:{},norm_J:{}'.format(iters,norm_J))
                success=True
                break
            data['J_rms']=norm_J/sqrt_N
            # data['rs']=norm_J
            if iters>0:
                Jdx=dot(J,dx)
                if Jdx<0:
                    denom=dot(dx,system.mul(data,dx))
                    dx=-Jdx/denom*dx
                else:
                    dx=torch.zeros_like(anchors_reg,dtype=anchors_reg.dtype,device=anchors_reg.device)

            # s=np.load('tmp/bad_s.npy')
            # s=torch.from_numpy(s).to(device=data['device'],dtype=data['dtype'])
            # print('inner:',system.inner(system.mul(data,s),s).item())
            # assert(False)

            dx,cg_iters=cg_solve(system,data,dx,-J,tol=self.cg_tol,max_iterations=self.cg_max_iter)

            def f(x):
                data=system.get_data(x)
                data['anchors_reg']=anchors_reg
                data['anchors_net']=anchors_net
                data['stiffen_anchors_reg']=stiffen_anchors_reg
                data['stiffen_anchors_net']=stiffen_anchors_reg
                J=system.get_J(data)
                return torch.norm(J)
            alpha,x=backtrack_line_search(x,dx,J,f,norm_J,alpha=0,max_steps=8)            

            # data['vts']=x
            # energy=system.compute_edge_energy(data)+system.compute_anchors_net_energy(data)
            # print('iter:{},energy:{},norm_J:{},cg_iters:{},alpha:{}'.format(iters,energy,norm_J,cg_iters,alpha))
            if alpha==0:
                print('no further process possible')
                break
            iters+=1
            if iters>=max_iter:
                print('reach max iter')
                break
        return x,data,success

def spring_opt_backward(system,data,dx,cg_tol,cg_max_iter):
    dc=torch.zeros_like(dx,device=dx.device,dtype=dx.dtype)
    dc,_=cg_solve(system,data,dc,dx,tol=cg_tol,max_iterations=cg_max_iter)
    dc*=(data['stiffen_anchors_net']+data['stiffen_anchors_reg'])
    return dc

from collections import defaultdict

def gather_data(data_list):
    tensor_list=defaultdict(list)
    for data in data_list:
        for k,v in data.items():
            if k=='edge_clamp_map':
                continue
            if type(v).__name__.find('Tensor')<0:
                continue
            tensor_list[k].append(v.unsqueeze(0))
    data_dict={}
    for k,l in tensor_list.items():
        data_dict[k]=torch.cat(l,dim=0)
    return data_dict

def scatter_data(data_dict):
    data_list=None
    for k,v in data_dict.items():
        if data_list is None:
            data_list=[{} for i in range(len(v))]
        for i in range(len(v)):
            data_list[i][k]=v[i]
    return data_list

def save_data(data_dir,data_dict):
    for k,v in data_dict.items():
        path=join(data_dir,'{}.npy'.format(k))
        np.save(path,v.cpu().numpy())

def load_data(data_dir,device):
    files=os.listdir(data_dir)
    data_dict={}
    for file in files:
        k=file[:-4]
        path=join(data_dir,file)
        v=np.load(path)
        v=torch.from_numpy(v).to(device)
        data_dict[k]=v
    return data_dict
######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import torch.nn as nn
from cg_solve import cg_solve
import math

def get_inner_mask(mask):
    sum_mask=torch.zeros_like(mask,device=mask.device,dtype=mask.dtype)
    d=1
    sum_mask[d:,:]+=mask[:-d,:]
    sum_mask[:-d,:]+=mask[d:,:]
    sum_mask[:,d:]+=mask[:,:-d]
    sum_mask[:,:-d]+=mask[:,d:]
    return (sum_mask>3.5).to(device=mask.device,dtype=mask.dtype)

class ImgOptSystem:
    # use HxWxD layout
    def __init__(self,mask,l0,lambda0=0):
        self.mask=mask
        self.inner_mask=get_inner_mask(mask).unsqueeze(2)
        self.mask_sum=torch.sum(self.inner_mask)
        self.use_edge=True
        self.use_bend=True
        self.use_cross=True
        self.use_gravity=False
        self.use_LM=False
        if self.use_edge:
            self.edge_slice_x=[slice(None),slice(1,None),slice(None),slice(None,-1)]
            self.edge_mask_x=self.get_mask(mask,self.edge_slice_x)
            self.edge_slice_y=[slice(1,None),slice(None),slice(None,-1),slice(None)]
            self.edge_mask_y=self.get_mask(mask,self.edge_slice_y)
            self.edge_slice=[slice(None),slice(1,-1),slice(1,-1),slice(None)]
            self.edge_l0_x=l0
            self.edge_l0_y=l0
            self.edge_k=10/(1+math.sqrt(2))
            # self.edge_k=1
        if self.use_bend:
            self.bend_slice_x=[slice(None),slice(2,None),slice(None),slice(None,-2)]
            self.bend_mask_x=self.get_mask(mask,self.bend_slice_x)
            self.bend_slice_y=[slice(2,None),slice(None),slice(None,-2),slice(None)]
            self.bend_mask_y=self.get_mask(mask,self.bend_slice_y)
            self.bend_slice=[slice(None),slice(2,-2),slice(2,-2),slice(None)]
            self.bend_l0_x=2*l0
            self.bend_l0_y=2*l0
            # self.bend_k=2/(1+math.sqrt(2))
            self.bend_k=0
        if self.use_cross:
            self.cross_slice_l=[slice(1,None),slice(1,None),slice(None,-1),slice(None,-1)]
            self.cross_mask_l=self.get_mask(mask,self.cross_slice_l)
            self.cross_slice_r=[slice(1,None),slice(None,-1),slice(None,-1),slice(1,None)]
            self.cross_mask_r=self.get_mask(mask,self.cross_slice_r)
            self.cross_slice=[slice(1,-1),slice(1,-1),slice(1,-1),slice(1,-1)]
            self.cross_l0_l=math.sqrt(2)*l0
            self.cross_l0_r=math.sqrt(2)*l0
            # self.cross_k=10/(1+math.sqrt(2))
            self.cross_k=0
        if self.use_gravity:
            self.density=5
            m=self.density*l0*l0
            g=9.8
            self.mg=m*g
        self.lambda0=lambda0


    def get_mask(self,mask,s):
        return ((mask[s[0],s[1]]+mask[s[2],s[3]])>1.5).to(dtype=mask.dtype).unsqueeze(2)

    def compute_data(self,img,mask,s,eps=1e-8):
        d=img[s[0],s[1],:]-img[s[2],s[3],:]
        d*=mask
        l=torch.norm(d,dim=2,keepdim=True)
        l=torch.clamp(l,min=eps)
        lhat=d/l
        return d,l,lhat

    def get_edge_data(self,img):
        d_x,l_x,lhat_x=self.compute_data(img,self.edge_mask_x,self.edge_slice_x)
        d_y,l_y,lhat_y=self.compute_data(img,self.edge_mask_y,self.edge_slice_y)
        return d_x,l_x,lhat_x,d_y,l_y,lhat_y

    def get_bend_data(self,img):
        d_x,l_x,lhat_x=self.compute_data(img,self.bend_mask_x,self.bend_slice_x)
        d_y,l_y,lhat_y=self.compute_data(img,self.bend_mask_y,self.bend_slice_y)
        return d_x,l_x,lhat_x,d_y,l_y,lhat_y

    def get_cross_data(self,img):
        d_l,l_l,lhat_l=self.compute_data(img,self.cross_mask_l,self.cross_slice_l)
        d_r,l_r,lhat_r=self.compute_data(img,self.cross_mask_r,self.cross_slice_r)
        return d_l,l_l,lhat_l,d_r,l_r,lhat_r

    def get_data(self,img):
        data={}
        data['size']=img.size()
        data['device']=img.device
        data['dtype']=img.dtype
        if self.use_edge:
            data['edge']=self.get_edge_data(img)
        if self.use_bend:
            data['bend']=self.get_bend_data(img)
        if self.use_cross:
            data['cross']=self.get_cross_data(img)
        return data

    def add_J(self,data,l0,mask,J,s,Js,alpha=1):
        d,l,lhat=data
        j=(l-l0)*d
        j=j/l
        j*=mask
        J[Js[0],Js[1],:]+=(-j[s[0],s[1],:]+j[s[2],s[3],:])*alpha

    def add_edge_J(self,data,J):
        self.add_J(data[:3],self.edge_l0_x,self.edge_mask_x,J,self.edge_slice_x,self.edge_slice[:2],self.edge_k)
        self.add_J(data[3:],self.edge_l0_y,self.edge_mask_y,J,self.edge_slice_y,self.edge_slice[2:],self.edge_k)

    def add_bend_J(self,data,J):
        self.add_J(data[:3],self.bend_l0_x,self.bend_mask_x,J,self.bend_slice_x,self.bend_slice[:2],self.bend_k)
        self.add_J(data[3:],self.bend_l0_y,self.bend_mask_y,J,self.bend_slice_y,self.bend_slice[2:],self.bend_k)

    def add_cross_J(self,data,J):
        self.add_J(data[:3],self.cross_l0_l,self.cross_mask_l,J,self.cross_slice_l,self.cross_slice[:2],self.cross_k)
        self.add_J(data[3:],self.cross_l0_r,self.cross_mask_r,J,self.cross_slice_r,self.cross_slice[2:],self.cross_k)

    def get_J(self,data):
        J=torch.zeros(data['size'],device=data['device'],dtype=data['dtype'])
        if self.use_edge:
            self.add_edge_J(data['edge'],J)
        if self.use_bend:
            self.add_bend_J(data['bend'],J)
        if self.use_cross:
            self.add_cross_J(data['cross'],J)
        if self.lambda0!=0:
            # if there is only 1 newton step, no need to do anything here
            pass
        if self.use_gravity:
            J[:,:,1]+=-self.mg

        J*=self.inner_mask

        return J

    def add_Hu(self,data,u,l0,mask,Hu,s,Hus,alpha=1):
        d,l,lhat=data
        du=u[s[0],s[1],:]-u[s[2],s[3],:]
        d2u=(l-l0)*du+l0*(lhat*torch.sum(lhat*du,dim=2,keepdim=True))
        d2u=d2u/l
        d2u=d2u*mask
        Hu[Hus[0],Hus[1],:]+=(-d2u[s[0],s[1],:]+d2u[s[2],s[3],:])*alpha

    def add_edge_Hu(self,data,u,Hu):
        self.add_Hu(data[:3],u,self.edge_l0_x,self.edge_mask_x,Hu,self.edge_slice_x,self.edge_slice[:2],self.edge_k)
        self.add_Hu(data[3:],u,self.edge_l0_y,self.edge_mask_y,Hu,self.edge_slice_y,self.edge_slice[2:],self.edge_k)

    def add_bend_Hu(self,data,u,Hu):
        self.add_Hu(data[:3],u,self.bend_l0_x,self.bend_mask_x,Hu,self.bend_slice_x,self.bend_slice[:2],self.bend_k)
        self.add_Hu(data[3:],u,self.bend_l0_y,self.bend_mask_y,Hu,self.bend_slice_y,self.bend_slice[2:],self.bend_k)

    def add_cross_Hu(self,data,u,Hu):
        self.add_Hu(data[:3],u,self.cross_l0_l,self.cross_mask_l,Hu,self.cross_slice_l,self.cross_slice[:2],self.cross_k)
        self.add_Hu(data[3:],u,self.cross_l0_r,self.cross_mask_r,Hu,self.cross_slice_r,self.cross_slice[2:],self.cross_k)

    def matmul(self,data,u):
        Hu=torch.zeros_like(u,device=u.device,dtype=u.dtype)
        if self.use_edge:
            self.add_edge_Hu(data['edge'],u,Hu)
        if self.use_bend:
            self.add_bend_Hu(data['bend'],u,Hu)
        if self.use_cross:
            self.add_cross_Hu(data['cross'],u,Hu)
        if self.lambda0!=0 and not self.use_LM:
            Hu+=u*self.lambda0
        if self.use_gravity:
            # gravity is not in hessian
            pass

        Hu*=self.inner_mask

        return Hu

    def mul(self,data,u):
        if self.use_LM:
            Hu=self.matmul(data,self.matmul(data,u))
            Hu+=self.lambda0*u
            Hu*=self.inner_mask
            return Hu
        else:
            return self.matmul(data,self.matmul(data,u))

    def inner(self,x,y):
        return torch.sum(x*y)

    def convergence_norm(self,r):
        # return torch.norm(r*r)/self.mask_sum
        return torch.max(torch.norm(r,dim=2))

    def get_energy(self,l,mask,l0):
        return torch.sum((l[mask.byte()]-l0)**2)

    def get_edge_energy(self,img):
        img=img.permute(1,2,0)
        _,l_x,_,_,l_y,_=self.get_edge_data(img)
        return self.get_energy(l_x,self.edge_mask_x,self.edge_l0_x)+self.get_energy(l_y,self.edge_mask_y,self.edge_l0_y)

    def get_bend_energy(self,img):
        img=img.permute(1,2,0)
        _,l_x,_,_,l_y,_=self.get_bend_data(img)
        return self.get_energy(l_x,self.bend_mask_x,self.bend_l0_x)+self.get_energy(l_y,self.bend_mask_y,self.bend_l0_y)

    def get_cross_energy(self,img):
        img=img.permute(1,2,0)
        _,l_l,_,_,l_r,_=self.get_cross_data(img)
        return self.get_energy(l_l,self.cross_mask_l,self.cross_l0_l)+self.get_energy(l_r,self.cross_mask_r,self.cross_l0_r)

class NewtonStepper:
    def __init__(self,mask,l0,lambda0=0,cg_tol=1e-3,cg_max_iter=1000):
        self.system=ImgOptSystem(mask,l0,lambda0=lambda0)
        self.cg_tol=cg_tol
        self.cg_max_iter=cg_max_iter

    def step(self,img,img0=None):
        system=self.system
        img=img.permute(1,2,0)
        data=system.get_data(img)
        J=system.get_J(data)
        HJ=system.matmul(data,J)
        dx0=torch.zeros_like(img,dtype=img.dtype,device=img.device)
        # solve HTH dx = HJ
        dx,cg_iters=cg_solve(system,data,dx0,HJ,tol=self.cg_tol,max_iterations=self.cg_max_iter)
        return (img-dx).permute(2,0,1),cg_iters if cg_iters>0 else self.cg_max_iter



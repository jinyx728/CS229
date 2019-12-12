######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
from collections import defaultdict
import torch
from debug_utils import print_tensors
import torch.nn.functional as F
import torch.nn as nn
class GANLossManager:
    def __init__(self,ctx):
        self.clear()
        self.l1_fn=None

    def add_item_loss(self,name,v,num_samples):
        self.item_loss[name]+=v
        self.item_samples[name]+=num_samples

    def clear(self):
        self.item_loss=defaultdict(float)
        self.item_samples=defaultdict(float)

    def get_item_loss(self):
        item_loss={name:loss/self.item_samples[name] for name,loss in self.item_loss.items()}
        return item_loss

    def masked_select(self,x,masks):
        n_samples=len(x)
        result=[]
        for i in range(n_samples):
            result.append(x[i][masks[i]].unsqueeze(0))
        return torch.cat(result,dim=0)

    def gradient_penalty(self, y, x, masks):
        weights = torch.ones(y.size()).to(y.device)
        dydx = torch.autograd.grad(
            y, x, grad_outputs=weights, retain_graph=True, create_graph=True)[0]
        # dydx=self.masked_select(dydx,masks)
        dydx=dydx*masks
        dydx = dydx.view(dydx.size(0), -1)
        # print('dydx','min',torch.min(dydx).item(),'max',torch.max(dydx).item(),'mean',torch.mean(dydx).item())
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm - 1)**2)

    def get_loss_D(self,netD,offset_imgs):
        return torch.mean(netD(offset_imgs))

    # def get_loss_D(self,netD,pd_offset_imgs,gt_offset_imgs):
    #     loss_pd=torch.mean(netD(pd_offset_imgs))
    #     loss_gt=torch.mean(netD(gt_offset_imgs))
    #     loss_D=loss_pd-loss_gt

    #     return loss_D
    def get_loss_G(self,netD,pd_offset_imgs):
        return -torch.mean(netD(pd_offset_imgs))

    def get_loss_gp(self,netD,pd_offset_imgs,gt_offset_imgs,masks):
        n_samples=len(pd_offset_imgs)
        t=torch.rand(n_samples,1,1,1,device=gt_offset_imgs.device,dtype=gt_offset_imgs.dtype)
        xhat = (1 - t) * gt_offset_imgs + t * pd_offset_imgs
        xhat=xhat.detach()
        xhat.requires_grad_(True)
        yhat = netD(xhat)
        loss_gp = self.gradient_penalty(yhat, xhat, masks)
        return loss_gp

    def get_loss_R1(self,netD,x,masks=None,y=None):
        x.requires_grad_(True)
        if y is None:
            y=netD(x)
        weights = torch.ones(y.size()).to(y.device)
        dydx = torch.autograd.grad(
            y, x, grad_outputs=weights, retain_graph=True, create_graph=True, only_inputs=True)[0]
        # dydx=self.masked_select(dydx,masks)
        if masks is not None:
            dydx=dydx*masks
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sum(dydx**2, dim=1)
        # x.requires_grad_(False)
        return torch.mean(dydx_l2norm)

    def get_loss_l1(self,pd_offset_imgs,gt_offset_imgs,masks):
        return torch.sum(torch.abs((pd_offset_imgs-gt_offset_imgs)*masks))/torch.sum(masks)

    def get_vt_loss(self,pd_vt,gt_vt):
        assert(pd_vt.size(2)==3)
        assert(gt_vt.size(2)==3)
        return torch.mean(torch.norm(pd_vt-gt_vt,dim=2))

    def get_cat_loss_D(self,raw,label):
        device=raw.device
        dtype=raw.dtype
        if label=='real':
            if not hasattr(self,'real_onehot'):
                self.real_onehot=torch.tensor([1,0,0],dtype=dtype,device=device,requires_grad=False).view(1,3)
            tgt=self.real_onehot
        elif label=='sim':
            if not hasattr(self,'sim_onehot'):
                self.sim_onehot=torch.tensor([0,1,0],dtype=dtype,device=device,requires_grad=False).view(1,3)
            tgt=self.sim_onehot
        elif label=='fake':
            if not hasattr(self,'fake_onehot'):
                self.fake_onehot=torch.tensor([0,0,1],dtype=dtype,device=device,requires_grad=False).view(1,3)
            tgt=self.fake_onehot
        else:
            print('unrecognized label')
        # src=F.softmax(raw,dim=1)
        src=raw
        if not hasattr(self,'bce_loss'):
            self.bce_loss=nn.BCEWithLogitsLoss()
        tgt=tgt.repeat(src.size(0),1)
        return self.bce_loss(src,tgt)

    def get_cat_loss_G(self,raw,lambda_ctgr_sim):
        return torch.mean(raw[:,0]-raw[:,2]+lambda_ctgr_sim*(raw[:,1]-raw[:,2]))

    def get_sqr_loss(self,netD,imgs,tgt):
        y=netD(imgs)
        return torch.mean((netD(imgs)-tgt)**2)

    def get_proj_loss_D(self,netD,offset_imgs,cond):
        real,cond=netD(offset_imgs,cond)
        return torch.mean(real),torch.mean(cond)

    def get_proj_loss_R1(self,loss_D,offset_imgs,cond,masks):
        img_R1=self.get_loss_R1(None,offset_imgs,masks,y=loss_D)
        cond_R1=self.get_loss_R1(None,cond,y=loss_D)
        return img_R1+cond_R1

    def get_proj_loss_G(self,netD,offset_imgs,cond):
        real,cond=netD(offset_imgs,cond)
        return -torch.mean(real),-torch.mean(cond)

    def get_cond_loss_D(self,netD,offset_imgs,cond):
        return torch.mean(netD(offset_imgs,cond))

    def get_cond_loss_G(self,netD,offset_imgs,cond):
        return -torch.mean(netD(offset_imgs,cond))

    def get_consensus_loss(self,net,loss):
        dydx=torch.autograd.grad(loss,net.parameters(),retain_graph=True,create_graph=True,allow_unused=True)
        s=0
        for d in dydx:
            if d is not None:
                s+=torch.sum(d**2)

        return s

    def get_cond_loss_R1(self,netD,offset_imgs,cond,masks):
        x=offset_imgs
        x.requires_grad_(True)
        y=netD(x,cond)
        weights = torch.ones(y.size()).to(y.device)
        dydx = torch.autograd.grad(
            y, x, grad_outputs=weights, retain_graph=True, create_graph=True, only_inputs=True)[0]
        if masks is not None:
            dydx=dydx*masks
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sum(dydx**2, dim=1)
        # x.requires_grad_(False)
        return torch.mean(dydx_l2norm)

    def get_cond_loss_R2(self,netD,offset_imgs,cond,masks):
        return self.get_cond_loss_R1(netD,offset_imgs,cond,masks)

    def get_cond_loss_zgp(self,netD,gt_offset_imgs,pd_offset_imgs,cond,masks):
        n_samples=len(pd_offset_imgs)
        t=torch.rand(n_samples,1,1,1,device=gt_offset_imgs.device,dtype=gt_offset_imgs.dtype)
        xhat = (1 - t) * gt_offset_imgs + t * pd_offset_imgs
        xhat=xhat.detach()
        return self.get_cond_loss_R1(netD,xhat,cond,masks)


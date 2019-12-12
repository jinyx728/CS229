######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Train_Conv_Mix')
from inequality_opt import InequalitySolver
import torch
import numpy as np
import os
from os.path import join
from timeit import print_stat,clear_stat
from obj_io import Obj,read_obj,write_obj
from cvx_opt import CvxOpt
from cvx_opt_func import cvx_opt 
# from multiprocessing import Pool
import torch.multiprocessing as mp
# torch.multiprocessing.set_start_method('spawn')

class SimpleInequalityOptTest:
    def __init__(self):
        self.dtype=torch.double
        self.device=torch.device("cuda:0")
    def test(self):
        m=torch.tensor([2,1],device=self.device,dtype=self.dtype).view(-1,1)
        edges=torch.tensor([0,1],device=self.device,dtype=torch.long).view(1,-1)
        l0=torch.tensor([1],device=self.device,dtype=self.dtype).view(-1,1)
        x=torch.tensor([[0,0,0],
                        [2,0,0]],device=self.device,dtype=self.dtype)
        solver=InequalitySolver(m,edges,l0)
        print(solver.solve(x))

class InequalityOptTest:
    def __init__(self):
        self.shared_data_dir='../../shared_data_midres'
        self.data_root_dir='/data/zhenglin/poses_v3'
        self.gt_dir='opt_test'
        self.pd_dir='opt_test'
        self.cr_dir='opt_test'

        self.device=torch.device('cuda:0')
        self.dtype=torch.double

        linear_edges=np.loadtxt(join(self.shared_data_dir,'linear_edges.txt')).astype(int)
        linear_edges=torch.from_numpy(linear_edges).to(device=self.device,dtype=torch.long)
        linear_rest_lengths=np.loadtxt(join(self.shared_data_dir,'mat_or_med_linear.txt'))
        linear_rest_lengths=torch.from_numpy(linear_rest_lengths).to(device=self.device,dtype=self.dtype)
        bend_edges=np.loadtxt(join(self.shared_data_dir,'bend_edges.txt')).astype(int)
        bend_edges=torch.from_numpy(bend_edges).to(device=self.device,dtype=torch.long)
        bend_rest_lengths=np.loadtxt(join(self.shared_data_dir,'mat_or_med_bend.txt'))
        bend_rest_lengths=torch.from_numpy(bend_rest_lengths).to(device=self.device,dtype=self.dtype)
        self.edges=torch.cat([linear_edges,bend_edges],dim=0)
        self.l0=torch.cat([linear_rest_lengths,bend_rest_lengths]).view(-1,1)
        
        # max_num_constraints=2
        max_num_constraints=-1
        if max_num_constraints>=0:
            self.edges=self.edges[:max_num_constraints]
            self.l0=self.l0[:max_num_constraints]

        tshirt_obj_path=os.path.join(self.shared_data_dir,'flat_tshirt.obj')
        tshirt_obj=read_obj(tshirt_obj_path)
        n_vts=len(tshirt_obj.v)
        self.m=torch.ones((n_vts,1)).to(device=self.device,dtype=self.dtype)
        self.solver=InequalitySolver(self.m,self.edges,self.l0)

    def get_err(self,gt_vt,pd_vt):
        err=torch.sum((gt_vt-pd_vt)**2,dim=1,keepdim=True)*self.system.m
        return torch.sqrt(torch.sum(err)/torch.sum(self.system.m)).item()

    def test_opt(self,sample_id,save_obj=False):
        pd_obj=read_obj(join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id)))
        pd_vt=torch.from_numpy(pd_obj.v).to(device=self.device,dtype=self.dtype)
        cr_vt=self.solver.solve(pd_vt)
        if save_obj:
            cr_vt=cr_vt.cpu().numpy()
            cr_obj=Obj(v=cr_vt,f=pd_obj.f)
            write_obj(cr_obj,join(self.cr_dir,'cr_{:08d}.obj'.format(sample_id)))

    def test_func(self,sample_id,save_obj=False):
        n_workers=2

        pd_obj=read_obj(join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id)))
        pd_vt=torch.from_numpy(pd_obj.v).to(device=self.device,dtype=self.dtype)
        pd_vts=pd_vt.unsqueeze(0).repeat(n_workers,1,1)
        pd_vts.requires_grad_(True)

        gt_obj=read_obj(join(self.gt_dir,'gt_{:08d}.obj'.format(sample_id)))
        gt_vt=torch.from_numpy(gt_obj.v).to(device=self.device,dtype=self.dtype)
        gt_vts=gt_vt.unsqueeze(0).repeat(n_workers,1,1)

        n_vts=len(gt_vt)
        edges=self.edges.cpu().numpy()
        l0=self.l0.view(-1).cpu().numpy()

        pool=mp.Pool(n_workers)
        opts=pool.starmap(CvxOpt,[(n_vts,edges,l0)]*n_workers)

        cr_vts=cvx_opt(pd_vts,opts,pool,self.solver.system)
        loss_fn=torch.nn.MSELoss()
        loss=loss_fn(cr_vts,gt_vts)
        loss.backward()
        print('grad_norm',torch.norm(pd_vts.grad).item())

        # test=torch.autograd.gradcheck(cvx_opt,(cr_vts,opts,pool,self.solver.system,self.solver.get_data),eps=1e-6,atol=1e-4)
        # numerical check
        dv=torch.rand(pd_vts.size()).to(device=self.device,dtype=self.dtype)
        dv=(dv*2-1)*1e-4
        pd_vts_p=pd_vts+dv
        cr_vts_p=cvx_opt(pd_vts_p,opts,pool,self.solver.system)
        loss_p=loss_fn(cr_vts_p,gt_vts)
        loss_n=loss+torch.sum(pd_vts.grad*dv)
        err=(loss_n-loss_p)/loss_p
        print('loss',loss.item(),'loss_p',loss_p.item(),'err',err.item())

if __name__=='__main__':
    # test=SimpleInequalityOptTest()
    # test.test()
    # torch.multiprocessing.set_start_method('spawn')

    test=InequalityOptTest()
    test.test_opt(106,save_obj=True)
    # test.test_func(106)
    # print_stat()
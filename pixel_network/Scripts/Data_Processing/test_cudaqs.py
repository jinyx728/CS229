######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Train_Conv_Mix')
import torch
import numpy as np
import os
from os.path import join,isdir,isfile
from obj_io import Obj,read_obj,write_obj
from timeit import timeit
from patch_utils import PatchManager
import argparse
import matplotlib.pyplot as plt
import time
from cudaqs_func import CudaqsModule,init_cudaqs_module


class CudaqsTest:
    def __init__(self):
        self.data_root_dir='/data/zhenglin/poses_v3'
        self.pd_dir='spring_test/cudaqs'
        self.cr_dir='spring_test/cudaqs'
        self.shared_data_dir='../../shared_data'

        self.device=torch.device('cuda:0')
        self.dtype=torch.double

        self.patch_id=-1
        # self.patch_id=13

        self.batch_size=1
        self.use_variable_m=True

        res_ctx={'shared_data_dir':self.shared_data_dir}
        ctx={'batch_size':self.batch_size,'dtype':self.dtype,'device':self.device,'patch_id':self.patch_id,'max_num_constraints':-1,'verbose':True,'use_variable_m':self.use_variable_m,'stiffen_anchor_factor':0.2}
        self.res_ctx=res_ctx
        self.ctx=ctx
        self.module=init_cudaqs_module(res_ctx,ctx)
        self.stiffen_anchor=self.module.stiffen_anchor

        flat_obj=read_obj(join(self.shared_data_dir,'flat_tshirt.obj'))
        self.f=flat_obj.f
        self.agg_n_vts=len(flat_obj.v)

        if self.patch_id>=0:
            self.pd_dir=join(self.pd_dir,'p{}'.format(self.patch_id))
            self.cr_dir=join(self.cr_dir,'p{}'.format(self.patch_id))
        else:
            self.pd_dir=join(self.pd_dir,'whole')
            self.cr_dir=join(self.cr_dir,'whole')

        if not isdir(self.pd_dir):
            os.makedirs(self.pd_dir)
        if not isdir(self.cr_dir):
            os.makedirs(self.cr_dir)

    def read_obj(self,path,patch_id=-1):
        pd_obj=read_obj(path)
        pd_vt=pd_obj.v
        f=pd_obj.f
        if patch_id>=0:
            pd_vt=pd_vt[self.patch_vt_ids]
            f=f[self.patch_fc_ids]
        return pd_vt,f

    def write_obj(self,v,f,out_path,patch_id=-1):
        if patch_id>=0:
            full_v=np.zeros((self.agg_n_vts,3))
            full_v[self.patch_vt_ids,:]=v
            v=full_v
        print('write to',out_path)
        write_obj(Obj(v=v,f=f),out_path)

    def test_forward(self,sample_id):
        pd_path=join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id))
        v,f=self.read_obj(pd_path,patch_id=self.patch_id)      
        v=torch.from_numpy(v).to(device=self.device,dtype=self.dtype)
        v=v.unsqueeze(0).repeat(self.batch_size,1,1)
        anchor=v

        start_time=time.time()
        # print('stiffen_anchor:',stiffen_anchor.size())
        x=self.module(anchor)
        end_time=time.time()
        print('module takes:{}s'.format(end_time-start_time))

        x_save=x[self.batch_size-1].detach().cpu().numpy()
        cr_path=join(self.cr_dir,'cr_{:08d}.obj'.format(sample_id))
        self.write_obj(x_save,f,cr_path,patch_id=self.patch_id)

    def test_backward(self,sample_id):
        pd_path=join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id))
        pd_v,f=self.read_obj(pd_path,patch_id=self.patch_id)
        pd_v=torch.from_numpy(pd_v).to(device=self.device,dtype=self.dtype)
        pd_v=pd_v.unsqueeze(0).repeat(self.batch_size,1,1)
        pd_v.requires_grad_(True)
        gt_path=join(self.pd_dir,'gt_{:08d}.obj'.format(sample_id))
        gt_v,_=self.read_obj(gt_path,patch_id=self.patch_id)
        gt_v=torch.from_numpy(gt_v).to(device=self.device,dtype=self.dtype)
        gt_v=gt_v.unsqueeze(0).repeat(self.batch_size,1,1)
        cr_v=self.module(pd_v,self.stiffen_anchor)
        loss=torch.sum((gt_v-cr_v)**2)/2
        loss.backward()
        print('grad.norm',torch.norm(pd_v.grad).item())
        grad_path=join(self.cr_dir,'grad_{:08d}.npy'.format(sample_id))
        print('save to',grad_path)
        np.save(grad_path,pd_v.grad[0].detach().cpu().numpy())

    def test_grad(self,sample_id):
        pd_path=join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id))
        print('pd_path',pd_path)
        pd_vt,_=self.read_obj(pd_path,patch_id=self.patch_id)
        grad=np.load(join(self.cr_dir,'grad_{:08d}.npy'.format(sample_id)))
        grad_len=1
        ed_vt=pd_vt-grad*grad_len
        n_vts=len(pd_vt)
        obj_path=join(self.cr_dir,'grad_{:08d}.obj'.format(sample_id))
        print('write to',obj_path)
        with open(obj_path,'w') as f:
            for v in pd_vt:
                f.write('v {} {} {}\n'.format(v[0],v[1],v[2]))
            for v in ed_vt:
                f.write('v {} {} {}\n'.format(v[0],v[1],v[2]))
            for i in range(n_vts):
                f.write('l {} {}\n'.format(i+1,i+1+n_vts))

    def test_samples_dir(self,samples_dir):
        sample_dirs=os.listdir(samples_dir)
        pd_pattern='pd_cloth_{:08d}.obj'
        cr_pattern='cr_cloth_{:08d}.obj'
        for sample_dir in sample_dirs:
            sample_id=int(sample_dir)
            pd_path=join(samples_dir,sample_dir,pd_pattern.format(sample_id))
            pd_obj=read_obj(pd_path)
            v,f=pd_obj.v,pd_obj.f
            v=torch.from_numpy(v).to(device=self.device,dtype=self.dtype)
            v=v.unsqueeze(0)
            anchor=v
            x=self.module(anchor)
            x_save=x[0].detach().cpu().numpy()
            cr_path=join(samples_dir,sample_dir,cr_pattern.format(sample_id))
            print('write to',cr_path)
            write_obj(Obj(v=x_save,f=f),cr_path)

if __name__=='__main__':
    test=CudaqsTest()
    # test.test_forward(106)
    test.test_backward(106)
    # test.test_grad(106)
    # test.test_samples_dir('../../rundir/lowres_vt/uvn/eval_test')
    # test.test_samples_dir('../../rundir/lowres_vt/uvn/eval_train')

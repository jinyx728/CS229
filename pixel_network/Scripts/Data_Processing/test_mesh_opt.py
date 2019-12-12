######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Train_Conv_Mix')
import os
from os.path import join,isfile,isdir
import gzip
from mesh_opt import NewtonStepper
import time
from obj_io import read_obj
import numpy as np
import torch
import matplotlib.pyplot as plt
from offset_img_utils import OffsetManager
from offset_io_utils import OffsetIOManager
from PIL import Image
import pickle as pk
from collections import defaultdict

def load_np_img(path):
    if path.endswith('.gz'):
        with gzip.open(path,'rb') as f:
            arr=np.load(file=f)
    else:
        arr=np.load(path)
    return torch.from_numpy(arr).permute(2,0,1)

def save_png_img(path,img):
    if img.ndim==2:
        Image.fromarray(np.uint8(img*255)).save(path)


class MeshOptTest:
    def __init__(self,lambda0=0,cg_tol=1e-3,cg_max_iter=1000,n_newton_steps=1,beta=0.7):
        self.shared_data_dir='../../shared_data_midres'
        self.data_root_dir='/data/zhenglin/poses_v3'
        self.res=128
        self.lambda0=lambda0
        self.cg_tol=cg_tol
        self.cg_max_iter=cg_max_iter
        self.n_newton_steps=n_newton_steps
        self.beta=beta
        self.pd_offset_dir='opt_test'
        self.gt_offset_dir=join(self.data_root_dir,'midres_offset_npys')
        self.skin_vt_dir=join(self.data_root_dir,'midres_skin_npys')
        self.cr_vt_dir='opt_test'
        self.out_obj_dir='opt_test'

        self.device=torch.device('cuda:0')
        self.dtype=torch.double

        self.offset_manager=OffsetManager(self.shared_data_dir,ctx={'data_root_dir':self.data_root_dir,'offset_img_size':self.res,'device':self.device})
        self.offset_io_manager=OffsetIOManager(res_ctx={'skin_dir':None,'shared_data_dir':self.shared_data_dir})

        tshirt_obj_path=os.path.join(self.shared_data_dir,'flat_tshirt.obj')
        tshirt_obj=read_obj(tshirt_obj_path)
        self.rest_vts=torch.from_numpy(tshirt_obj.v).to(device=self.device,dtype=self.dtype)
        self.fcs=torch.from_numpy(tshirt_obj.f).to(device=self.device,dtype=torch.long)

        front_vt_ids=np.loadtxt(join(self.shared_data_dir,'front_vertices.txt')).astype(np.int32)
        back_vt_ids=np.loadtxt(join(self.shared_data_dir,'back_vertices.txt')).astype(np.int32)
        self.stepper=NewtonStepper(self.rest_vts,self.fcs,front_vt_ids=front_vt_ids,back_vt_ids=back_vt_ids,bdry_ids=None,lambda0=lambda0,cg_tol=self.cg_tol,cg_max_iter=self.cg_max_iter)
        self.system=self.stepper.system

        self.verbose=True

    def get_err(self,gt_vt,pd_vt):
        err=torch.sum((gt_vt-pd_vt)**2,dim=1,keepdim=True)*self.system.m
        return torch.sqrt(torch.sum(err)/torch.sum(self.system.m)).item()

    def get_egy(self,vt):
        return self.system.get_total_egy(vt).item()

    def test(self,sample_id,save_obj=False):
        skin_vt_path=join(self.skin_vt_dir,'skin_{:08d}.npy'.format(sample_id))
        skin_vt=torch.from_numpy(np.load(skin_vt_path)).to(device=self.device,dtype=self.dtype)
        pd_img_path=join(self.pd_offset_dir,'pd_img_{:08d}.npy.gz'.format(sample_id))
        pd_img=load_np_img(pd_img_path).to(self.device,dtype=self.dtype)
        pd_offset=self.offset_manager.get_offsets_from_offset_imgs_both_sides(pd_img.unsqueeze(0)).squeeze().to(dtype=self.dtype)
        pd_vt=pd_offset+skin_vt
        gt_offset_path=join(self.gt_offset_dir,'offset_{:08d}.npy'.format(sample_id))
        gt_offset=torch.from_numpy(np.load(gt_offset_path)).to(self.device,dtype=self.dtype)
        gt_vt=gt_offset+skin_vt

        cr_vt=pd_vt
        start_time=time.time()
        with torch.no_grad():
            lambda0=self.lambda0
            for i in range(self.n_newton_steps):
                self.stepper.system.lambda0=lambda0
                cr_vt,cg_iters=self.stepper.step(cr_vt,vt0=pd_vt)
                lambda0*=self.beta
        end_time=time.time()

        total_time=end_time-start_time

        pd_err=self.get_err(pd_vt,gt_vt)
        cr_err=self.get_err(cr_vt,gt_vt)

        gt_egy=self.get_egy(gt_vt)
        pd_egy=self.get_egy(pd_vt)
        cr_egy=self.get_egy(cr_vt)

        if save_obj:
            obj_path=join(self.out_obj_dir,'cr_{:08d}.obj'.format(sample_id))
            self.offset_io_manager.write_cloth(cr_vt.cpu().numpy(),obj_path)

        data={'time':total_time,'iters':cg_iters}
        data['err_cr']=cr_err
        data['err_pd']=pd_err
        data['egy_gt']=gt_egy
        data['egy_pd']=pd_egy
        data['egy_cr']=cr_egy
        return data

def get_data_str(data):
    s=[]
    items=sorted(data.items(),key=lambda v:v[0])
    for k,v in items:
        if type(v)==float:
            s.append('{}:{:.8e}'.format(k,v))
        else:
            s.append('{}:{}'.format(k,v))
    return ','.join(s)

def test_lambda0(vs,samples_dir,sample_list,out_dir):
    save_data={}
    for lambda0 in vs:
        print('lambda0:',lambda0)
        test=MeshOptTest(lambda0=lambda0,cg_tol=1e-4,cg_max_iter=1000)
        agg_data=defaultdict(list)
        for sample_id in sample_list:
            print('sample_id',sample_id)
            test.pd_offset_dir=join(samples_dir,'{:08d}'.format(sample_id))
            test_data=test.test(sample_id)
            for k,v in test_data.items():
                agg_data[k].append(v)
            print(get_data_str(test_data))
        save_data[lambda0]=agg_data
    if not isdir(out_dir):
        os.makedirs(out_dir)
    out_path=join(out_dir,'data.pk')
    with open(out_path,'wb') as fout:
        pk.dump(save_data,fout)

def vlz_lambda0(out_dir):
    save_data_path=join(out_dir,'data.pk')
    with open(save_data_path,'rb') as fin:
        save_data=pk.load(fin)
    avg_data=defaultdict(list)
    for lambda0,agg_data in save_data.items():
        for k,v in agg_data.items():
            # if type(v[0])!=float:
            #     avg_data[k].append((lambda0,torch.mean(torch.tensor(v).float()).item()))
            # else:
            avg_data[k].append((lambda0,np.mean(np.array(v))))
    # time 
    fig=plt.figure()
    ax=plt.gca()
    l=sorted(avg_data['time'],key=lambda v:v[0])[2:]
    X,Y=map(list,zip(*l))
    ax.plot(X,Y)
    fig.savefig(join(out_dir,'time.png'))

    fig=plt.figure()
    ax=plt.gca()
    l=sorted(avg_data['err_pd'],key=lambda v:v[0])[1:]
    X,Y=map(list,zip(*l))
    ax.plot(X,Y,label='pd')
    l=sorted(avg_data['err_cr'],key=lambda v:v[0])[1:]
    X,Y=map(list,zip(*l))
    ax.plot(X,Y,label='cr')
    ax.legend()
    fig.savefig(join(out_dir,'err.png'))

    fig=plt.figure()
    ax=plt.gca()
    l=sorted(avg_data['egy_pd'],key=lambda v:v[0])[1:]
    X,Y=map(list,zip(*l))
    ax.plot(X.copy(),Y.copy(),label='pd')
    l=sorted(avg_data['egy_cr'],key=lambda v:v[0])[1:]
    X,Y=map(list,zip(*l))
    ax.plot(X.copy(),Y.copy(),label='cr')
    ax.legend()
    fig.savefig(join(out_dir,'egy.png'))

def test_edge_stats(samples_dir,sample_list,out_dir):
    test=MeshOptTest(cg_tol=1e-4,cg_max_iter=1000)
    agg_data=defaultdict(list)
    for sample_id in sample_list:
        print('sample_id',sample_id)
        skin_vt_path=join(test.skin_vt_dir,'skin_{:08d}.npy'.format(sample_id))
        skin_vt=torch.from_numpy(np.load(skin_vt_path)).to(device=test.device,dtype=test.dtype)
        pd_offset_dir=join(samples_dir,'{:08d}'.format(sample_id))
        pd_img_path=join(pd_offset_dir,'pd_img_{:08d}.npy.gz'.format(sample_id))
        pd_img=load_np_img(pd_img_path).to(test.device,dtype=test.dtype)
        pd_offset=test.offset_manager.get_offsets_from_offset_imgs_both_sides(pd_img.unsqueeze(0)).squeeze().to(dtype=test.dtype)
        pd_vt=pd_offset+skin_vt
        gt_offset_path=join(test.gt_offset_dir,'offset_{:08d}.npy'.format(sample_id))
        gt_offset=torch.from_numpy(np.load(gt_offset_path)).to(test.device,dtype=test.dtype)
        gt_vt=gt_offset+skin_vt
        system=test.system

        gt_linear_ratios=system.get_lengths(gt_vt,system.linear_edges)/system.linear_rest_lengths
        gt_linear_ratios=gt_linear_ratios.squeeze().cpu().numpy().tolist()
        gt_bend_ratios=system.get_lengths(gt_vt,system.bend_edges)/system.bend_rest_lengths
        gt_bend_ratios=gt_bend_ratios.squeeze().cpu().numpy().tolist()
        agg_data['gt']+=gt_linear_ratios+gt_bend_ratios

        pd_linear_ratios=system.get_lengths(pd_vt,system.linear_edges)/system.linear_rest_lengths
        pd_linear_ratios=pd_linear_ratios.squeeze().cpu().numpy().tolist()
        pd_bend_ratios=system.get_lengths(pd_vt,system.bend_edges)/system.bend_rest_lengths
        pd_bend_ratios=pd_bend_ratios.squeeze().cpu().numpy().tolist()
        agg_data['pd']+=pd_linear_ratios+pd_bend_ratios

    save_data=agg_data
    if not isdir(out_dir):
        os.makedirs(out_dir)
    out_path=join(out_dir,'data.pk')
    with open(out_path,'wb') as fout:
        pk.dump(save_data,fout)

def vlz_edge_stats(out_dir):
    save_data_path=join(out_dir,'data.pk')
    with open(save_data_path,'rb') as fin:
        save_data=pk.load(fin)

    # gt 
    fig=plt.figure()
    ax=plt.gca()
    ax.hist(save_data['gt'],range=[0.9,1.1],bins=20)
    ax.plot([1,1],[0,1300000])
    fig.savefig(join(out_dir,'gt.png'))

    fig=plt.figure()
    ax=plt.gca()
    ax.hist(save_data['pd'],range=[0.9,1.1],bins=20)
    ax.plot([1,1],[0,300000])
    fig.savefig(join(out_dir,'pd.png'))

def test_newton_steps(ns,samples_dir,sample_list,out_dir):
    save_data={}
    for n in ns:
        print('n:',n)
        test=MeshOptTest(lambda0=1.6e7,cg_tol=1e-4,cg_max_iter=1000,n_newton_steps=n,beta=0.5)
        agg_data=defaultdict(list)
        for sample_id in sample_list:
            print('sample_id',sample_id)
            test.pd_offset_dir=join(samples_dir,'{:08d}'.format(sample_id))
            test_data=test.test(sample_id)
            for k,v in test_data.items():
                agg_data[k].append(v)
            print(get_data_str(test_data))
        save_data[n]=agg_data
    if not isdir(out_dir):
        os.makedirs(out_dir)
    out_path=join(out_dir,'data.pk')
    with open(out_path,'wb') as fout:
        pk.dump(save_data,fout)

def vlz_newton_steps(out_dir):

    save_data_path=join(out_dir,'data.pk')
    with open(save_data_path,'rb') as fin:
        save_data=pk.load(fin)
    avg_data=defaultdict(list)
    for n,agg_data in save_data.items():
        for k,v in agg_data.items():
            avg_data[k].append((n,np.mean(np.array(v))))
    # time 
    fig=plt.figure()
    ax=plt.gca()
    l=sorted(avg_data['time'],key=lambda v:v[0])
    l.pop(2)
    X,Y=map(list,zip(*l))
    ax.plot(X,Y)
    fig.savefig(join(out_dir,'time.png'))

    fig=plt.figure()
    ax=plt.gca()
    l=sorted(avg_data['err_pd'],key=lambda v:v[0])
    l.pop(2)
    X,Y=map(list,zip(*l))
    ax.plot(X,Y,label='pd')
    l=sorted(avg_data['err_cr'],key=lambda v:v[0])
    l.pop(2)
    X,Y=map(list,zip(*l))
    ax.plot(X,Y,label='cr')
    ax.legend()
    fig.savefig(join(out_dir,'err.png'))

    fig=plt.figure()
    ax=plt.gca()
    l=sorted(avg_data['egy_pd'],key=lambda v:v[0])
    l.pop(2)
    X,Y=map(list,zip(*l))
    ax.plot(X.copy(),Y.copy(),label='pd')
    l=sorted(avg_data['egy_cr'],key=lambda v:v[0])
    l.pop(2)
    X,Y=map(list,zip(*l))
    ax.plot(X.copy(),Y.copy(),label='cr')
    ax.legend()
    fig.savefig(join(out_dir,'egy.png'))


if __name__=='__main__':
    # sample_list_path='/data/zhenglin/poses_v3/sample_lists/midres_test_samples.txt'
    # out_dir='opt_test/mesh_lambda0'
    # test_lambda0([2e6,4e6,8e6,1.6e7,3.2e7,6.4e7,1.28e8],'../../rundir/midres/xyz/eval_test',np.loadtxt(sample_list_path)[1:101].astype(np.int32),out_dir)

    # sample_list_path='/data/zhenglin/poses_v3/sample_lists/midres_train_samples.txt'
    # out_dir='opt_test/mesh_lambda0_train'
    # test_lambda0([2e6,4e6,8e6,1.6e7,3.2e7,6.4e7,1.28e8],'../../rundir/midres/xyz/eval_train',np.loadtxt(sample_list_path)[:100].astype(np.int32),out_dir)

    # vlz_lambda0(out_dir)
    
    # test=MeshOptTest(lambda0=1.6e7,cg_tol=1e-4,cg_max_iter=1000,n_newton_steps=8,beta=0.7)
    # data=test.test(106,save_obj=True)
    # print(get_data_str(data))

    # sample_list_path='/data/zhenglin/poses_v3/sample_lists/midres_train_samples.txt'
    # out_dir='opt_test/mesh_edges_train'
    # test_edge_stats('../../rundir/midres/xyz/eval_train',np.loadtxt(sample_list_path)[:100].astype(np.int32),out_dir)

    # vlz_edge_stats(out_dir)

    # sample_list_path='/data/zhenglin/poses_v3/sample_lists/midres_train_samples.txt'
    out_dir='opt_test/mesh_step_train'
    # out_dir='opt_test/mesh_step_train_b0.5'
    # test_newton_steps([1,4,6,8,10],'../../rundir/midres/xyz/eval_train',np.loadtxt(sample_list_path)[:100].astype(np.int32),out_dir)

    vlz_newton_steps(out_dir)






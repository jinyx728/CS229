######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Train_Conv_Mix')
import os
from os.path import join,isfile,isdir
import gzip
from img_opt import NewtonStepper
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

def get_total_error(pd,gt,mask):
    d=(pd-gt)*mask.unsqueeze(0)
    return torch.sum(d*d)

def get_l0(shared_data_dir,H):
    l0_file=join(shared_data_dir,'l0_{}.txt'.format(H))
    if not isfile(l0_file):
        front_obj_path=join(shared_data_dir,'flat_tshirt_front.obj')
        front_obj=read_obj(front_obj_path)
        front_vts,front_fcs=front_obj.v,front_obj.f

        vts=front_vts
        xyzmin,xyzmax=np.min(vts,axis=0),np.max(vts,axis=0)
        ymin,ymax=xyzmin[1],xyzmax[1]
        ymin-=0.1
        ymax+=0.1
        l0=(ymax-ymin)/H
        print('write to',l0_file,'l0=',l0)
        np.savetxt(l0_file,np.array([l0]))
        return l0
    else:
        return float(np.loadtxt(l0_file))

def save_png_img(path,img):
    if img.ndim==2:
        Image.fromarray(np.uint8(img*255)).save(path)

def from_offset_img_to_rgb_img(img,mask,img_stats={'min':-0.1,'max':0.1}):
    img=(img-img_stats['min'])/(img_stats['max']-img_stats['min'])
    img=np.clip(img*255,0,255)
    return np.flip(np.uint8(img*mask),axis=0)

def save_offset_img(path,img,mask,img_stats={'min':-0.1,'max':0.1}):
    front_mask=mask[0,:,:].unsqueeze(2).cpu().numpy()
    front_offset_img=img[:3,:,:].permute(1,2,0).cpu().numpy()
    front_rgb_img=from_offset_img_to_rgb_img(front_offset_img,front_mask,img_stats=img_stats)

    back_mask=mask[1,:,:].unsqueeze(2).cpu().numpy()
    back_offset_img=img[3:,:,:].permute(1,2,0).cpu().numpy()
    back_rgb_img=from_offset_img_to_rgb_img(back_offset_img,back_mask,img_stats=img_stats)

    Image.fromarray(np.concatenate([front_rgb_img,back_rgb_img],axis=1)).save(path)

def get_img_stats(img):
    x,y,z=img[:,:,0],img[:,:,1],img[:,:,2]
    y=y[y>0.2]
    miny,maxy=np.min(y),np.max(y)
    dy=maxy-miny
    minx,maxx=np.min(x),np.max(x)
    minz,maxz=np.min(z),np.max(z)
    cx,cy,cz=((minx+maxx)/2),((miny+maxy)/2),((minz+maxz)/2)
    return {'min':np.array([cx-dy/2,miny,cz-dy/2]),'max':np.array([cx+dy/2,maxy,cz+dy/2])}

class ImgOptTest:
    def __init__(self,lambda0=0,cg_tol=1e-3):
        self.shared_data_dir='../../shared_data_midres'
        self.data_root_dir='/data/zhenglin/poses_v3'
        self.res=128
        self.lambda0=lambda0
        self.cg_tol=cg_tol
        self.cg_max_iter=1000
        self.skin_img_dir=join(self.data_root_dir,'midres_skin_imgs_{}'.format(self.res))
        self.pd_img_dir='opt_test'
        self.gt_img_dir=join(self.data_root_dir,'midres_offset_imgs_{}'.format(128))
        self.cr_img_dir='opt_test'
        self.out_obj_dir='opt_test'

        self.l0=get_l0(self.shared_data_dir,128)

        self.device=torch.device('cuda:0')
        self.dtype=torch.double
        mask_path=join(self.shared_data_dir,'offset_img_mask_no_pad_{}.npy'.format(self.res))
        self.mask=load_np_img(mask_path).to(self.device,dtype=self.dtype)

        self.offset_manager=OffsetManager(self.shared_data_dir,ctx={'data_root_dir':self.data_root_dir,'offset_img_size':self.res,'device':self.device})
        self.offset_io_manager=OffsetIOManager(res_ctx={'skin_dir':None,'shared_data_dir':self.shared_data_dir})

        self.front_mask=self.mask[0,:,:]
        self.front_sum=torch.sum(self.front_mask).item()
        self.back_mask=self.mask[1,:,:]
        self.back_sum=torch.sum(self.back_mask).item()
        self.front_stepper=NewtonStepper(self.front_mask,self.l0,lambda0=self.lambda0,cg_tol=self.cg_tol,cg_max_iter=self.cg_max_iter)
        self.back_stepper=NewtonStepper(self.back_mask,self.l0,lambda0=self.lambda0,cg_tol=self.cg_tol,cg_max_iter=self.cg_max_iter)

        self.verbose=True

    def get_err(self,front_gt_img,front_pd_img,back_gt_img,back_pd_img):
        front_pd_err=get_total_error(front_pd_img,front_gt_img,self.front_mask).item()
        back_pd_err=get_total_error(back_pd_img,back_gt_img,self.back_mask).item()
        pd_err=(front_pd_err+back_pd_err)/(self.front_sum+self.back_sum)
        return pd_err

    def get_egy(self,front_pd_img,back_pd_img):
        front_system=self.front_stepper.system
        front_edge_egy=front_system.get_edge_energy(front_pd_img).item()
        front_bend_egy=front_system.get_bend_energy(front_pd_img).item()
        front_cross_egy=front_system.get_cross_energy(front_pd_img).item()
        back_system=self.back_stepper.system
        back_edge_egy=back_system.get_edge_energy(back_pd_img).item()
        back_bend_egy=back_system.get_bend_energy(back_pd_img).item()
        back_cross_egy=back_system.get_cross_energy(back_pd_img).item()
        edge_egy,bend_egy,cross_egy=front_edge_egy+back_edge_egy,front_bend_egy+back_bend_egy,front_cross_egy+back_cross_egy
        return front_system.edge_k*edge_egy+front_system.bend_k*bend_egy+front_system.cross_k*cross_egy 

    def test(self,sample_id,save_obj=False,save_diff=False):
        pd_img_path=join(self.pd_img_dir,'pd_img_{:08d}.npy.gz'.format(sample_id))
        pd_img=load_np_img(pd_img_path).to(self.device,dtype=self.dtype)
        gt_img_path=join(self.gt_img_dir,'offset_img_{:08d}.npy'.format(sample_id))
        gt_img=load_np_img(gt_img_path).to(self.device,dtype=self.dtype)
        skin_img_path=join(self.skin_img_dir,'skin_img_{:08d}.npy.gz'.format(sample_id))
        skin_img=load_np_img(skin_img_path).to(self.device,dtype=self.dtype)
        pd_img+=skin_img
        gt_img+=skin_img

        cr_img_path=join(self.cr_img_dir,'cr_img_{:08d}.npy'.format(sample_id))

        front_pd_img=pd_img[:3,:,:]
        front_gt_img=gt_img[:3,:,:]
        back_pd_img=pd_img[3:,:,:]
        back_gt_img=gt_img[3:,:,:]
        front_cr_img=front_pd_img
        back_cr_img=back_pd_img

        start_time=time.time()
        with torch.no_grad():
            front_cr_img,front_iters=self.front_stepper.step(front_cr_img)
            back_cr_img,back_iters=self.back_stepper.step(back_cr_img)
        end_time=time.time()

        total_time=end_time-start_time

        pd_err=self.get_err(front_gt_img,front_pd_img,back_gt_img,back_pd_img)
        cr_err=self.get_err(front_gt_img,front_cr_img,back_gt_img,back_cr_img)

        pd_egy=self.get_egy(front_pd_img,back_pd_img)
        cr_egy=self.get_egy(front_cr_img,back_cr_img)

        if save_obj:
            cr_img=torch.cat([front_cr_img,back_cr_img],dim=0)
            cr_vts=self.offset_manager.get_offsets_from_offset_imgs_both_sides(cr_img.unsqueeze(0))
            obj_path=join(self.out_obj_dir,'cr_{:08d}.obj'.format(sample_id))
            self.offset_io_manager.write_cloth(cr_vts[0].cpu().numpy(),obj_path)
            if save_diff:
                save_offset_img(join(self.out_obj_dir,'cr_{:08d}.png'.format(sample_id)),cr_img-pd_img,self.mask,img_stats={'min':-1e-3,'max':1e-3})

        data={'time':total_time,'iters_front':front_iters,'iters_back':back_iters,'err_cr':cr_err,'egy_cr':cr_egy}
        data['err_pd']=pd_err
        data['egy_pd']=pd_egy
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
        test=ImgOptTest(lambda0=lambda0)
        agg_data=defaultdict(list)
        for sample_id in sample_list:
            print('sample_id',sample_id)
            test.pd_img_dir=join(samples_dir,'{:08d}'.format(sample_id))
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
    l=sorted(avg_data['pd_err'],key=lambda v:v[0])[2:]
    X,Y=map(list,zip(*l))
    ax.plot(X,Y,label='pd')
    l=sorted(avg_data['cr_err'],key=lambda v:v[0])[2:]
    X,Y=map(list,zip(*l))
    ax.plot(X,Y,label='cr')
    ax.legend()
    fig.savefig(join(out_dir,'err.png'))

    fig=plt.figure()
    ax=plt.gca()
    l=sorted(avg_data['pd_egy'],key=lambda v:v[0])[2:]
    X,Y=map(list,zip(*l))
    ax.plot(X,Y,label='pd')
    l=sorted(avg_data['cr_egy'],key=lambda v:v[0])[2:]
    X,Y=map(list,zip(*l))
    ax.plot(X,Y,label='cr')
    ax.legend()
    fig.savefig(join(out_dir,'egy.png'))


if __name__=='__main__':
    sample_list_path='/data/zhenglin/poses_v3/sample_lists/midres_test_samples.txt'
    out_dir='opt_test/test_lambda0'
    # test_lambda0([0,0.5,1,2,4,8,16,32,64],'../../rundir/midres/xyz/eval_test',np.loadtxt(sample_list_path)[:64].astype(np.int32),'opt_test/test_lambda0')
    # vlz_lambda0(out_dir)
    test=ImgOptTest(lambda0=32,cg_tol=1e-4)
    data=test.test(106,save_obj=True,save_diff=True)
    print(get_data_str(data))



######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Train_Conv_Mix')
import os
from os.path import join
import torch
import numpy as np
from render_offsets import get_vts_normalize_m,normalize_vts,load_crpds
from offset_img_utils import get_normalize_pos
from obj_io import Obj,read_obj,write_obj
from timeit import timeit
from PIL import Image


def barycentric_coord(location,x1,x2,x3):
    u=x2-x1
    v=x3-x1
    w=location-x1
    u_dot_u=np.inner(u,u)
    v_dot_v=np.inner(v,v)
    u_dot_v=np.inner(u,v)
    u_dot_w=np.inner(u,w)
    v_dot_w=np.inner(v,w)
    denominator=u_dot_u*v_dot_v-u_dot_v*u_dot_v
    if np.abs(denominator)>1e-16:
        one_over_denominator=1/denominator
    else:
        one_over_denominator=1e16
    a=(v_dot_v*u_dot_w-u_dot_v*v_dot_w)*one_over_denominator
    b=(u_dot_u*v_dot_w-u_dot_v*u_dot_w)*one_over_denominator
    return np.array([1-a-b,a,b])

def cvt_crpds(crpds):
    d={}
    for i in range(len(crpds)):
        p=crpds[i]
        d[p[1]]=p[0]
    return d

def save_png_img(path,img):
    img=np.clip((img*255),0,255).astype(np.uint8)
    print(img.dtype)
    Image.fromarray(img).save(path)

def bdot(a,b):
    return torch.sum(a*b,dim=2,keepdim=True)

def get_vt_ids_img(fc_id_img,fcs):
    print('fc_id_img',fc_id_img.size(),torch.min(fc_id_img).item(),torch.max(fc_id_img).item(),len(fcs))
    vt_ids_img=fcs[fc_id_img]
    print('vt_ids_img',vt_ids_img.size(),torch.min(vt_ids_img),torch.max(vt_ids_img))
    return vt_ids_img


def get_vt_ws_img(vt_ids_img,vts_2d,size):
    device,dtype=vts_2d.device,vts_2d.dtype
    def get_vt_img(vt_ids_img,vts_2d):
        img=vts_2d[vt_ids_img]
        return img
    def get_p(size):
        W,H=size
        x=torch.linspace(-1+1/W,1-1/W,W).view(1,-1,1).repeat(H,1,1).to(device=device,dtype=dtype)
        y=torch.linspace(-1+1/H,1-1/H,H).view(-1,1,1).repeat(1,W,1).to(device=device,dtype=dtype)
        return torch.cat([x,y],dim=2)

    a=get_vt_img(vt_ids_img[:,:,0],vts_2d)
    b=get_vt_img(vt_ids_img[:,:,1],vts_2d)
    c=get_vt_img(vt_ids_img[:,:,2],vts_2d)
    p=get_p(size)
    eps=1e-8
    scale_factor=1e2
    def get_uvw(a,b,c,p):
        v0=b-a
        v1=c-a
        v2=p-a
        # scale up for numerical stability
        v0*=scale_factor
        v1*=scale_factor
        v2*=scale_factor
        d00=bdot(v0,v0)
        d01=bdot(v0,v1)
        d11=bdot(v1,v1)
        d20=bdot(v2,v0)
        d21=bdot(v2,v1)
        denom=d00*d11-d01*d01
        denom[(denom>=0)*(denom<eps)]=eps
        denom[(denom<0)*(denom>-eps)]=-eps
        v=(d11*d20-d01*d21)/denom
        w=(d00*d21-d01*d20)/denom
        u=1-v-w
        uvw=torch.cat([u,v,w],dim=2)
        return uvw
    uvw=get_uvw(a,b,c,p)
    return uvw
    
class RasterizeUtils:
    def __init__(self):
        self.shared_data_dir='../../shared_data'
        # self.shared_data_dir='../../shared_data_midres'

        front_obj_path=os.path.join(self.shared_data_dir,'flat_tshirt_front.obj')
        front_obj=read_obj(front_obj_path)
        front_vts,front_fcs=front_obj.v,front_obj.f
        front_vts_normalize_m=get_vts_normalize_m(front_vts)

        front_obj_w_pad_path=os.path.join(self.shared_data_dir,'front_tshirt_w_pad.obj')
        front_obj_w_pad=read_obj(front_obj_w_pad_path)
        front_vts_w_pad,front_fcs_w_pad=front_obj_w_pad.v,front_obj_w_pad.f
        front_vts_w_pad=normalize_vts(front_vts_w_pad[:,:2],front_vts_normalize_m)

        back_obj_path=os.path.join(self.shared_data_dir,'flat_tshirt_back.obj')
        back_obj=read_obj(back_obj_path)
        back_vts,back_fcs=back_obj.v,back_obj.f
        back_vts_normalize_m=get_vts_normalize_m(back_vts)

        back_obj_w_pad_path=os.path.join(self.shared_data_dir,'back_tshirt_w_pad.obj')
        back_obj_w_pad=read_obj(back_obj_w_pad_path)
        back_vts_w_pad,back_fcs_w_pad=back_obj_w_pad.v,back_obj_w_pad.f
        back_vts_w_pad=normalize_vts(back_vts_w_pad[:,:2],back_vts_normalize_m)

        front_crpds_path=os.path.join(self.shared_data_dir,'front_vert_crpds.txt')
        front_crpds=cvt_crpds(load_crpds(front_crpds_path))
        back_crpds_path=os.path.join(self.shared_data_dir,'back_vert_crpds.txt')
        back_crpds=cvt_crpds(load_crpds(back_crpds_path))

        print('len(front)',len(front_vts),'len(back)',len(back_vts))
        self.front_vts=front_vts_w_pad
        self.back_vts=back_vts_w_pad
        self.front_fcs=front_fcs_w_pad
        self.back_fcs=back_fcs_w_pad
        self.front_crpds=front_crpds
        self.back_crpds=back_crpds
        self.img_dir='opt_test/rasterize'

    def proc_fcs(self,fcs,crpds):
        proc_fcs=fcs.copy()
        n_fcs=len(fcs)
        for fc_id in range(n_fcs):
            for i in range(3):
                v=proc_fcs[fc_id,i]
                if v in crpds:
                    proc_fcs[fc_id,i]=crpds[v]
        return proc_fcs

    def get_rasterize_info_gpu(self,vts,fcs,fc_id_img,crpds,img_size):
        vt_ids_img=get_vt_ids_img(fc_id_img,fcs)
        vt_ws_img=get_vt_ws_img(vt_ids_img,vts[:,:2],(img_size,img_size))
        proc_fcs=self.proc_fcs(fcs.cpu().numpy(),crpds)
        # update fc ids for crpds
        proc_fcs=torch.from_numpy(proc_fcs).to(fc_id_img.device,dtype=torch.long)
        vt_ids_img=get_vt_ids_img(fc_id_img,proc_fcs)
        return vt_ids_img,vt_ws_img

    def rasterize_gpu(self,img_size,device=torch.device("cuda:0")):
        mask=np.load(join(self.shared_data_dir,'offset_img_mask_{}.npy'.format(img_size)))
        fc_id_img=np.load(join(self.shared_data_dir,'fc_id_img_{}.npy'.format(img_size)))
        fc_id_img=torch.from_numpy(fc_id_img).to(device,dtype=torch.long)
        front_vts=torch.from_numpy(self.front_vts).to(device)
        front_fcs=torch.from_numpy(self.front_fcs).to(device,dtype=torch.long)
        front_fc_id_img=fc_id_img[:,:,0]
        front_vt_ids_img,front_vt_ws_img=self.get_rasterize_info_gpu(front_vts,front_fcs,front_fc_id_img,self.front_crpds,img_size)
        front_vt_ids_img,front_vt_ws_img=front_vt_ids_img.cpu().numpy(),front_vt_ws_img.cpu().numpy()

        back_vts=torch.from_numpy(self.back_vts).to(device)
        back_fcs=torch.from_numpy(self.back_fcs).to(device,dtype=torch.long)
        back_fc_id_img=fc_id_img[:,:,3]
        back_vt_ids_img,back_vt_ws_img=self.get_rasterize_info_gpu(back_vts,back_fcs,back_fc_id_img,self.back_crpds,img_size)
        back_vt_ids_img,back_vt_ws_img=back_vt_ids_img.cpu().numpy(),back_vt_ws_img.cpu().numpy()

        front_mask=mask[:,:,:1]
        back_mask=mask[:,:,1:2]

        # print('front_mask.unique',np.unique(front_mask))
        # print('back_mask.unique',np.unique(front_mask))
        front_vt_ids_img=(front_vt_ids_img*front_mask).astype(np.int)
        front_vt_ws_img=front_vt_ws_img*front_mask

        back_vt_ids_img=(back_vt_ids_img*back_mask).astype(np.int)
        back_vt_ws_img=back_vt_ws_img*back_mask
        vt_ids_img=np.concatenate([front_vt_ids_img,back_vt_ids_img],axis=2)
        vt_ws_img=np.concatenate([front_vt_ws_img,back_vt_ws_img],axis=2)
        np.save(join(self.shared_data_dir,'vt_ids_img_{}.npy'.format(img_size)),vt_ids_img)
        np.save(join(self.shared_data_dir,'vt_ws_img_{}.npy'.format(img_size)),vt_ws_img)


        self.save_id_img(join(self.img_dir,'front_vt_0.png'),front_vt_ids_img[:,:,0],np.max(front_vt_ids_img),front_mask)
        self.save_id_img(join(self.img_dir,'front_vt_1.png'),front_vt_ids_img[:,:,1],np.max(front_vt_ids_img),front_mask)
        self.save_id_img(join(self.img_dir,'front_vt_2.png'),front_vt_ids_img[:,:,2],np.max(front_vt_ids_img),front_mask)
        self.save_ws_img(join(self.img_dir,'front_ws.png'),front_vt_ws_img,front_mask)

        print('back_vt_ids_img',np.min(back_vt_ids_img),np.max(back_vt_ids_img))

        self.save_id_img(join(self.img_dir,'back_vt_0.png'),back_vt_ids_img[:,:,0],np.max(back_vt_ids_img),back_mask)
        self.save_id_img(join(self.img_dir,'back_vt_1.png'),back_vt_ids_img[:,:,1],np.max(back_vt_ids_img),back_mask)
        self.save_id_img(join(self.img_dir,'back_vt_2.png'),back_vt_ids_img[:,:,2],np.max(back_vt_ids_img),back_mask)
        self.save_ws_img(join(self.img_dir,'back_ws.png'),back_vt_ws_img,back_mask)

    @timeit
    def get_rasterize_info_cpu(self,vts,fcs,img_size,crpds=None):
        H,W=img_size
        fc_ids=-np.ones((H,W,1)).astype(np.int32)
        vt_ids=-np.ones((H,W,3)).astype(np.int32)
        vt_ws=-np.ones((H,W,3))
        t=np.zeros((len(vts),3))
        t[:,:2]=vts
        vts=t
        for py in range(H//2,H):
            print('py',py)
            for px in range(W):
                nx,ny=get_normalize_pos(np.array([px,py]),W,H)
                v=np.array([nx,ny,0])
                n_fcs=len(fcs)
                for fc_id in range(n_fcs):
                    fc=fcs[fc_id]
                    i0,i1,i2=fc[0],fc[1],fc[2]
                    v0,v1,v2=vts[i0],vts[i1],vts[i2]
                    w0,w1,w2=barycentric_coord(v,v0,v1,v2)
                    if w0>=0 and w1>=0 and w2>=0:
                        fc_ids[py,px]=fc_id
                        if crpds is not None:
                            v0=crpds[i0] if i0 in crpds else i0
                            v1=crpds[i1] if i1 in crpds else i1
                            v2=crpds[i2] if i2 in crpds else i2
                        vt_ids[py,px,:]=np.array([i0,i1,i2])
                        vt_ws[py,px,:]=np.array([w0,w1,w2])
            break
        return fc_ids,vt_ids,vt_ws

    def rasterize_cpu(self,img_size):
        mask=np.load(join(self.shared_data_dir,'offset_img_mask_{}.npy'.format(img_size)))
        front_mask=mask[:,:,:1]
        front_fc_ids,front_vt_ids,front_vt_ws=self.get_rasterize_info_cpu(self.front_vts,self.front_fcs,(img_size,img_size),self.front_crpds)
        np.save(join(self.shared_data_dir,'fc_ids.npy'),front_fc_ids)
        np.save(join(self.shared_data_dir,'vt_ids.npy'),front_vt_ids)
        np.save(join(self.shared_data_dir,'vt_ws.npy'),front_vt_ws)
        front_fc_ids[front_fc_ids<0]=0
        front_vt_ids[front_vt_ids<0]=0
        front_vt_ws[front_vt_ws<0]=0
        self.save_id_img(join(self.img_dir,'front_fc_id.png'),front_fc_ids,len(self.front_fcs),front_mask)
        self.save_id_img(join(self.img_dir,'front_vt_0.png'),front_vt_ids[:,:,0],self.n_vts,front_mask)
        self.save_id_img(join(self.img_dir,'front_vt_1.png'),front_vt_ids[:,:,1],self.n_vts,front_mask)
        self.save_id_img(join(self.img_dir,'front_vt_2.png'),front_vt_ids[:,:,2],self.n_vts,front_mask)
        self.save_ws_img(join(self.img_dir,'front_ws.png'),front_vt_ws,front_mask)

    def save_id_img(self,path,id_img,id_max,mask,random=False):
        if random:
            colors=np.random.rand(id_max,3)
            img=colors[id_img]
        else:
            img=id_img/id_max
            H,W=id_img.shape[0],id_img.shape[1]
            img=img.reshape((H,W,1))
        img=img*mask
        img=img[:,:,0]
        print('img.shape',img.shape)
        save_png_img(path,img)

    def save_ws_img(self,path,ws_img,mask):
        img=ws_img*mask
        save_png_img(path,img)

if __name__=='__main__':
    rasterizer=RasterizeUtils()
    rasterizer.rasterize_gpu(128)
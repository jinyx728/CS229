######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,isfile
import numpy as np
from obj_io import Obj, read_obj, write_obj
import torch
from torch.nn.functional import grid_sample

def get_pix_pos(vt,W,H):
    pix_pos=(vt+1)/2*np.array([W,H])-0.5
    return pix_pos

def get_normalize_pos(pix_pos,W,H):
    vt_pos=(pix_pos+0.5)/np.array([W,H])*2-1
    return vt_pos

class OffsetManager:
    def __init__(self,shared_data_dir,offset_img_size=None,use_torch=True,ctx=None):
        data_root_dir=ctx['data_root_dir']
        if offset_img_size is None:
            offset_img_size=ctx['offset_img_size']
        device=ctx['device']
        dtype=ctx['dtype']
        self.img_size=offset_img_size

        front_obj_path=os.path.join(shared_data_dir,'flat_tshirt_front.obj')
        front_obj=read_obj(front_obj_path)
        front_vts,front_fcs=front_obj.v,front_obj.f
        front_vts=self.normalize_vts(front_vts)
        front_vts=front_vts.astype(np.float32)
        front_vt_ids=np.loadtxt(os.path.join(shared_data_dir,'front_vertices.txt')).astype(np.int32)
        n_vts=len(front_vts)
        all_vts=front_vts[:,:2].copy()
        front_vts=front_vts[front_vt_ids,:2]

        back_obj_path=os.path.join(shared_data_dir,'flat_tshirt_back.obj')
        back_obj=read_obj(back_obj_path)
        back_vts,back_fcs=back_obj.v,back_obj.f
        back_vts=self.normalize_vts(back_vts)
        back_vts=back_vts.astype(np.float32)
        back_vt_ids=np.loadtxt(os.path.join(shared_data_dir,'back_vertices.txt')).astype(np.int32)
        back_vts=back_vts[back_vt_ids,:2]

        bdry_vt_ids_path=os.path.join(shared_data_dir,'bdry_vertices.txt')

        if not os.path.isfile(bdry_vt_ids_path):
            self.save_bdry_ids()
        bdry_vt_ids=np.loadtxt(bdry_vt_ids_path).astype(np.int32)

        vt_ids_img=None
        vt_ws_img=None
        vt_ids_img_path=join(shared_data_dir,'vt_ids_img_{}.npy'.format(offset_img_size))
        if isfile(vt_ids_img_path):
            vt_ids_img=np.load(vt_ids_img_path)
        vt_ws_img_path=join(shared_data_dir,'vt_ws_img_{}.npy'.format(offset_img_size))
        if isfile(vt_ws_img_path):
            vt_ws_img=np.load(vt_ws_img_path)

        mask=np.load(join(shared_data_dir,'offset_img_mask_{}.npy'.format(offset_img_size)))

        if use_torch:
            all_vts=torch.from_numpy(all_vts).to(device,dtype=dtype)
            front_vts=torch.from_numpy(front_vts).to(device,dtype=dtype)
            front_vt_ids=torch.from_numpy(front_vt_ids).to(device,dtype=torch.long)
            back_vts=torch.from_numpy(back_vts).to(device,dtype=dtype)
            back_vt_ids=torch.from_numpy(back_vt_ids).to(device,dtype=torch.long)
            bdry_vt_ids=torch.from_numpy(bdry_vt_ids).to(device,dtype=torch.long)
            if vt_ids_img is not None:
                vt_ids_img=torch.from_numpy(vt_ids_img).to(device,dtype=torch.long)
            if vt_ws_img is not None:
                vt_ws_img=torch.from_numpy(vt_ws_img).to(device,dtype=dtype)
            mask=torch.from_numpy(mask).to(device,dtype=dtype)
        
        self.device=device    
        self.data_root_dir=data_root_dir
        self.shared_data_dir=shared_data_dir
        self.all_vts=all_vts
        self.front_vts=front_vts
        self.front_vt_ids=front_vt_ids
        self.back_vts=back_vts
        self.back_vt_ids=back_vt_ids
        self.bdry_vt_ids=bdry_vt_ids
        self.fcs=np.concatenate([front_fcs,back_fcs],axis=0)
        self.n_vts=n_vts
        self.vt_ids_img=vt_ids_img
        self.vt_ws_img=vt_ws_img
        # print('front_vt_ws_img,unique',torch.unique(torch.sum(vt_ws_img[:,:,:3],dim=2)),vt_ws_img.size())
        # print('back_vt_ws_img,unique',torch.unique(torch.sum(vt_ws_img[:,:,3:],dim=2)),vt_ws_img.size())
        self.mask=mask
        self.mask_sum=torch.sum(mask)

    def normalize_vts(self,vts):
        xyzmin,xyzmax=np.min(vts,axis=0),np.max(vts,axis=0)
        ymin,ymax=xyzmin[1],xyzmax[1]
        ymin-=0.1
        ymax+=0.1
        xcenter=(xyzmin[0]+xyzmax[0])/2
        ycenter=(xyzmin[1]+xyzmax[1])/2
        normalized_vts=vts.copy()
        normalized_vts[:,0]=(vts[:,0]-xcenter)/(ymax-ymin)*2
        normalized_vts[:,1]=(vts[:,1]-ycenter)/(ymax-ymin)*2
        D=np.array([self.img_size,self.img_size,2])
        # normalized_vts=((normalized_vts+1)/2*(D-1)+0.5)/D*2-1
        normalized_vts=((normalized_vts+1)/2*D-0.5)/(D-1)*2-1
        normalized_vts[:,2]=0
        return normalized_vts

    def get_offsets_from_offset_imgs_both_sides(self,offset_imgs):
        D=offset_imgs.size(1)//2
        front_offset_imgs=offset_imgs[:,:D,:,:]
        front_offsets=self.get_offsets_from_offset_imgs(front_offset_imgs,self.front_vts)
        back_offset_imgs=offset_imgs[:,D:D*2,:,:]
        back_offsets=self.get_offsets_from_offset_imgs(back_offset_imgs,self.back_vts)
        offsets=self.merge_front_back_offsets(front_offsets,back_offsets)

        return offsets

    def get_offsets_from_offset_imgs(self,offset_imgs,vts):
        N=len(offset_imgs)
        device=offset_imgs.device
        D=offset_imgs.size(1)
        size=offset_imgs.size(2)
        # vts=((vts+1)/2*(size-1)+0.5)/size*2-1
        # vts=((vts+1)/2*size-0.5)/(size-1)*2-1

        vts=vts.type(offset_imgs.type()).to(device).view(1,1,vts.size(0),2).repeat(N,1,1,1)
        offsets=grid_sample(offset_imgs,vts,mode='bilinear')
        offsets=offsets.view(N,D,-1).permute(0,2,1)
        return offsets

    def get_offset_imgs_from_offsets(self,offsets,vt_ids_img,vt_ws_img,mask):
        n_samples=len(offsets)
        offset_imgs=[]
        vt_i0_img,vt_i1_img,vt_i2_img=vt_ids_img[:,:,0],vt_ids_img[:,:,1],vt_ids_img[:,:,2]
        vt_w0_img,vt_w1_img,vt_w2_img=vt_ws_img[:,:,:1],vt_ws_img[:,:,1:2],vt_ws_img[:,:,2:3]
        for i in range(n_samples):
            offset=offsets[i]
            v0,v1,v2=offset[vt_i0_img],offset[vt_i1_img],offset[vt_i2_img]
            offset_img=v0*vt_w0_img+v1*vt_w1_img+v2*vt_w2_img
            offset_img*=mask
            offset_imgs.append(offset_img.permute(2,0,1).unsqueeze(0))
        return torch.cat(offset_imgs,dim=0)

    def get_offset_imgs_from_offsets_both_sides(self,offsets):
        front_offset_imgs=self.get_offset_imgs_from_offsets(offsets,self.vt_ids_img[:,:,:3],self.vt_ws_img[:,:,:3],self.mask[:,:,:1])
        back_offset_imgs=self.get_offset_imgs_from_offsets(offsets,self.vt_ids_img[:,:,3:],self.vt_ws_img[:,:,3:],self.mask[:,:,1:])
        return torch.cat([front_offset_imgs,back_offset_imgs],dim=1)

    def get_offsets_from_uvn_offsets(self,uvn_offsets,uvn_hats):
        uvn_offsets=uvn_offsets.unsqueeze(3)
        return torch.matmul(uvn_hats,uvn_offsets).squeeze(3)

    def merge_front_back_offsets(self,front_offsets,back_offsets):
        N=len(front_offsets)
        device=front_offsets.device
        D=front_offsets.size(2)
        offsets=torch.zeros(N,self.n_vts,D).type(front_offsets.type()).to(device)
        front_vt_ids=self.front_vt_ids.to(device)
        back_vt_ids=self.back_vt_ids.to(device)
        bdry_vt_ids=self.bdry_vt_ids.to(device)

        front_vt_scatter_ids=front_vt_ids.view(1,-1,1).repeat(N,1,D)
        offsets=offsets.scatter_add(1,front_vt_scatter_ids,front_offsets)
        back_vt_scatter_ids=back_vt_ids.view(1,-1,1).repeat(N,1,D)
        offsets=offsets.scatter_add(1,back_vt_scatter_ids,back_offsets)
        bdry_vt_scatter_ids=bdry_vt_ids.view(1,-1,1).repeat(N,1,D)
        offsets=offsets.scatter(1,bdry_vt_scatter_ids,offsets[:,bdry_vt_ids,:]/2)


        # offsets[:,front_vt_ids,:]+=front_offsets
        # offsets[:,back_vt_ids,:]+=back_offsets
        # offsets[:,bdry_vt_ids,:]/=2

        return offsets

    def get_offsets_from_uvn_offset_imgs(self,uvn_offset_imgs,vts,uvn_hats):
        uvn_offsets=self.get_offsets_from_offset_imgs(uvn_offset_imgs,vts)
        offsets=self.get_offsets_from_uvn_offsets(uvn_offsets,uvn_hats)
        return offsets

    def get_offsets_from_uvn_offset_imgs_both_sides(self,uvn_offset_imgs,front_uvn_hats,back_uvn_hats):
        front_uvn_offset_imgs=uvn_offset_imgs[:,:3,:,:]
        front_uvn_hats=front_uvn_hats[:,self.front_vt_ids,:].type(uvn_offset_imgs.type())
        front_offsets=self.get_offsets_from_uvn_offset_imgs(front_uvn_offset_imgs,self.front_vts,front_uvn_hats)

        back_uvn_offset_imgs=uvn_offset_imgs[:,3:6,:,:]
        back_uvn_hats=back_uvn_hats[:,self.back_vt_ids,:].type(uvn_offset_imgs.type())
        back_offsets=self.get_offsets_from_uvn_offset_imgs(back_uvn_offset_imgs,self.back_vts,back_uvn_hats)

        offsets=self.merge_front_back_offsets(front_offsets,back_offsets)

        return offsets
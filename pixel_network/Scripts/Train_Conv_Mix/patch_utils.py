######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,exists
import numpy as np
from render_offsets import get_vts_normalize_m,normalize_vts
from obj_io import Obj,read_obj,write_obj
from offset_img_utils import get_pix_pos
from PIL import Image
class PatchManager:
    def __init__(self,shared_data_dir='../../shared_data'):
        # get vts
        obj_path=join(shared_data_dir,'flat_tshirt.obj')
        obj=read_obj(obj_path)
        vts=obj.v
        self.unnormalized_vts=vts[:,:2]
        m=get_vts_normalize_m(vts)
        vts=normalize_vts(vts[:,:2],m)
        self.vts=vts
        self.fcs=obj.f

        # get_patch_names
        self.patch_path=join(shared_data_dir,'regions')
        patch_names_path=join(self.patch_path,'region_names.txt')
        self.patch_names=self.load_patch_names(patch_names_path)
        self.n_patches=len(self.patch_names)
        print('num_patches',len(self.patch_names))
        self.img_size=512
        self.crop_size=512*5//16

        patch_crop_dir=join(self.patch_path,'region_crops')
        if not os.path.isdir(patch_crop_dir):
            os.mkdir(patch_crop_dir)
        self.patch_crop_dir=patch_crop_dir

        self.mask_margin_size=16
        mask_path=os.path.join(shared_data_dir,'offset_img_mask.npy')
        mask=np.load(mask_path)
        self.front_mask=mask[:,:,0]
        self.back_mask=mask[:,:,1]

    def load_patch_names(self,path):
        patch_names=[]
        with open(path) as f:
            while True:
                line=f.readline()
                if line=='':
                    break
                patch_names.append(line.rstrip())
        return patch_names

    def get_patch_crop(self,patch_id):
        print('patch_name',self.patch_names[patch_id])
        patch_name=self.patch_names[patch_id]
        patch_bdry_vt_ids=np.loadtxt(os.path.join(self.patch_path,'{}_boundary_vertices.txt'.format(patch_name))).astype(np.int32)-1
        patch_intr_vt_ids=np.loadtxt(os.path.join(self.patch_path,'{}_interior_vertices.txt'.format(patch_name))).astype(np.int32)-1
        patch_vt_ids=np.hstack([patch_bdry_vt_ids,patch_intr_vt_ids])
        patch_vts=self.vts[patch_vt_ids]
        max_xy=np.max(patch_vts,axis=0)
        min_xy=np.min(patch_vts,axis=0)
        size=max_xy-min_xy
        side='front' if patch_name.find('front')!=-1 else 'back'
        xcenter=(max_xy[0]+min_xy[0])/2
        ycenter=(max_xy[1]+min_xy[1])/2
        pix_pos=get_pix_pos(np.array([xcenter,ycenter]),W=self.img_size,H=self.img_size)
        x=int(pix_pos[0]-self.crop_size//2)
        y=int(pix_pos[1]-self.crop_size//2)

        return (x,y,self.crop_size,self.crop_size,side)

    def save_patch_crop(self,patch_id,crop):
        x,y,w,h,side=crop
        patch_name=self.patch_names[patch_id]
        crop_path=os.path.join(self.patch_crop_dir,'{}_crop.txt'.format(patch_name))
        with open(crop_path,'w') as f:
            f.write('{} {} {} {} {}'.format(x,y,w,h,side))

    def load_patch_crop(self,patch_path):
        with open(patch_path) as f:
            line=f.readline()
        parts=line.split()
        x,y,w,h,side=int(parts[0]),int(parts[1]),int(parts[2]),int(parts[3]),parts[4]
        return (x,y,w,h,side)

    def load_patch_vt_ids(self,patch_id):
        patch_name=self.patch_names[patch_id]
        patch_bdry_vt_ids=np.loadtxt(os.path.join(self.patch_path,'{}_boundary_vertices.txt'.format(patch_name))).astype(np.int32)-1
        patch_intr_vt_ids=np.loadtxt(os.path.join(self.patch_path,'{}_interior_vertices.txt'.format(patch_name))).astype(np.int32)-1
        patch_vt_ids=np.hstack([patch_bdry_vt_ids,patch_intr_vt_ids])
        return patch_vt_ids

    def get_patch_mask(self,patch_id,crop):
        x,y,w,h,side=crop
        patch_vt_ids=self.load_patch_vt_ids(patch_id)
        patch_vts=self.vts[patch_vt_ids]
        max_xy=np.max(patch_vts,axis=0)
        min_xy=np.min(patch_vts,axis=0)

        max_pix_pos=get_pix_pos(max_xy,W=self.img_size,H=self.img_size).astype(int)
        min_pix_pos=get_pix_pos(min_xy,W=self.img_size,H=self.img_size).astype(int)
        max_pix_pos+=self.mask_margin_size
        min_pix_pos-=self.mask_margin_size

        min_pix_pos[min_pix_pos<0]=0
        max_pix_pos[max_pix_pos>self.img_size]=self.img_size
        
        if side=='front':
            mask=self.front_mask
        else:
            mask=self.back_mask

        local_mask=np.zeros(mask.shape)
        local_mask[min_pix_pos[1]:max_pix_pos[1],min_pix_pos[0]:max_pix_pos[0]]=1
        mask=mask*local_mask

        return mask[y:y+h,x:x+w]

    def save_patch_mask(self,patch_id,mask,save_img=False):
        patch_name=self.patch_names[patch_id]
        mask_path=os.path.join(self.patch_crop_dir,'{}_mask.npy'.format(patch_name))
        np.save(mask_path,mask)
        if save_img:
            mask_img_path=os.path.join(self.patch_crop_dir,'{}_mask.png'.format(patch_name))
            Image.fromarray(np.uint8(mask*255)).save(mask_img_path)

    def get_patch_crop_path(self,test_id):
        patch_name=self.patch_names[test_id]
        return os.path.join(self.patch_crop_dir,'{}_crop.txt'.format(patch_name))

    def get_patch_mask_path(self,test_id):
        patch_name=self.patch_names[test_id]
        return os.path.join(self.patch_crop_dir,'{}_mask.npy'.format(patch_name))

    def convert_vts_to_crop_coord(self,vts,crop,original_size):
        x,y,w,h,_=crop
        original_W,original_H=original_size
        pix_pos=get_pix_pos(vts,W=original_W,H=original_H)
        new_vts=(pix_pos-np.array([x,y])+0.5)/np.array([w,h])*2-1
        return new_vts

    def get_vts_in_crop(self,vt_ids,crop,original_size):
        vts=self.unnormalized_vts[vt_ids]
        vts_in_crop=self.convert_vts_to_crop_coord(vts,crop,original_size)
        return vts_in_crop

    def crop_offset_img(self,offset_img,crop):
        x,y,w,h,side=crop
        if side=='front':
            side_slice=slice(0,3)
        elif side=='back':
            side_slice=slice(3,6)
        return offset_img[y:y+h,x:x+w,side_slice]

    def get_patch_edge_ids(self,patch_vt_ids,agg_edges):
        patch_vt_set=set(patch_vt_ids)
        patch_edge_ids=[]
        n_edges=len(agg_edges)
        for edge_id in range(n_edges):
            edge=agg_edges[edge_id]
            if all([e in patch_vt_set for e in edge]):            
                patch_edge_ids.append(edge_id)
        return patch_edge_ids

    def get_patch_edges(self,patch_id,agg_edges):
        patch_vt_ids=self.load_patch_vt_ids(patch_id)
        vt_ids_in_patch={patch_vt_ids[k]:k for k in range(len(patch_vt_ids))}
        patch_edge_ids=self.get_patch_edge_ids(patch_vt_ids,agg_edges)
        patch_edges=[]
        for edge_id in patch_edge_ids:
            edge=agg_edges[edge_id]
            # patch_edges.append([vt_ids_in_patch[edge[0]],vt_ids_in_patch[edge[1]]])
            patch_edges.append([vt_ids_in_patch[e] for e in edge])
        return np.array(patch_edges,dtype=np.int)

    def get_patch_fc_ids(self,patch_vt_ids,agg_fcs):
        patch_vt_set=set(patch_vt_ids)
        patch_fc_ids=[]
        n_fcs=len(agg_fcs)
        for fc_id in range(n_fcs):
            fc=agg_fcs[fc_id]
            if fc[0] in patch_vt_set and fc[1] in patch_vt_set and fc[2] in patch_vt_set:
                patch_fc_ids.append(fc_id)
        return patch_fc_ids

    def get_patch_local_fcs(self,patch_vt_ids,patch_global_fcs):
        global_to_local_vt_map={patch_vt_ids[local_id]:local_id for local_id in range(len(patch_vt_ids))}
        patch_local_fcs=np.zeros_like(patch_global_fcs)
        n_fcs=len(patch_global_fcs)
        for fc_id in range(n_fcs):
            for i in range(3):
                patch_local_fcs[fc_id][i]=global_to_local_vt_map[patch_global_fcs[fc_id][i]]
        return patch_local_fcs



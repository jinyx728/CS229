######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import os
from os.path import exists,join
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random
import gzip

def crop_offset_img(offset_img,crop):
    x,y,w,h,side=crop
    if side=='front':
        side_slice=slice(0,3)
    elif side=='back':
        side_slice=slice(3,6)
    return offset_img[y:y+h,x:x+w,side_slice]

class TshirtImageDataset(Dataset):
    """Tshirt deformation dataset represented as "images"."""

    def __init__(self,sample_list_file,res_ctx=None,ctx=None):
                    #regions related
                    # crop=None,crop_mask=None,vt_ids_in_crop=None,original_size=None):
        assert(res_ctx is not None)
        assert(ctx is not None)

        offset_img_dir=res_ctx['offset_img_dir']
        vt_offset_dir=res_ctx['vt_offset_dir']
        uvn_dir=res_ctx['uvn_dir']
        mask_file=res_ctx['mask_file']

        self.data_root_dir = ctx['data_root_dir']
        assert(exists(self.data_root_dir))

        self.rotation_mat_dir=ctx['rotation_mat_dir']
        assert(exists(self.rotation_mat_dir))

        max_num_samples=ctx['max_num_samples']
        print('sample_list_file',sample_list_file)
        assert(exists(sample_list_file))
        sample_list=np.loadtxt(sample_list_file).astype(np.int32)

        if max_num_samples>0:
            list_size=len(sample_list)
            self.sample_list=sample_list[:np.minimum(list_size,max_num_samples)]
        else:
            self.sample_list=sample_list
        self.num_samples=len(self.sample_list)

        exists(mask_file)
        self.mask=np.load(mask_file).astype(np.float64)

        if ctx['eval']=='none':
            self.mode='train'
        else:
            self.mode='eval'

        self.load_vt_offset=ctx['load_vt_offset']
        self.load_masks=False

        self.pd_only=ctx['pd_only']
        if not self.pd_only:
            self.offset_img_dir=offset_img_dir
            print('offset_img_dir',offset_img_dir)
            assert(exists(offset_img_dir))

            if self.load_vt_offset:
                self.vt_offset_dir=vt_offset_dir
                print('vt_offset_dir',vt_offset_dir)
                assert(exists(vt_offset_dir))
            self.is_patch_vt_offset=vt_offset_dir.find('pd')!=-1

            print('check offset_imgs and rotation mats')
            for i in range(self.num_samples):
                sample_id=self.sample_list[i]
                offset_img_file=join(offset_img_dir,'offset_img_{:08d}.npy.gz')
                exists(offset_img_file)
                rotation_file=join(self.rotation_mat_dir,'rotation_mat_{:08d}.npy')
                exists(rotation_file)

            print('check shapes')
            with gzip.open(join(self.offset_img_dir,'offset_img_{:08d}.npy.gz'.format(self.sample_list[0])),'rb') as f:
                offset_img0=np.load(file=f)
            offset_img_shape=offset_img0.shape
            assert(offset_img_shape[0]==self.mask.shape[0])
            assert(offset_img_shape[1]==self.mask.shape[1])
            assert((offset_img_shape[2]==4) if ctx['test_case']=='lowres_tex' or ctx['test_case']=='lowres_tex_vt' or ctx['test_case']=='highres_tex' else (offset_img_shape[2]==6))
            assert(self.mask.shape[2]==2)

            self.skin_dir=res_ctx['skin_dir']
            self.use_normals=ctx['use_normals']
            if self.use_normals:
                self.vn_dir=res_ctx['vn_dir']
                exists(self.vn_dir)
                exists(self.skin_dir)
            self.is_patch_skin=self.skin_dir.find('patch')!=-1

        self.use_uvn=ctx['use_uvn']
        if self.use_uvn:
            self.uvn_dir=uvn_dir
            exists(self.uvn_dir)

        self.use_patches=ctx['use_patches']
        if self.use_patches:
            crop=ctx['crop']
            crop_mask=ctx['crop_mask']
            vt_ids_in_crop=res_ctx['vt_ids_in_crop']
            original_size=ctx['original_size']

            self.crop=crop
            self.vt_ids_in_crop=vt_ids_in_crop

            assert(crop_mask is not None)
            self.crop_mask=crop_mask

        self.use_hero=ctx['use_hero']
        if self.use_hero:
            self.sim_dataset=None
            self.sim_ids=None

        self.load_skin_imgs=ctx['cat_skin_imgs'] and self.mode=='train'
        if self.load_skin_imgs:
            self.skin_img_dir=res_ctx['skin_img_dir']

        self.load_offset_img=True
        self.load_diff_img=False
        if ctx['use_diff']:
            self.diff_img_dir=res_ctx['diff_img_dir']
            if self.mode=='train':
                self.load_offset_img=False
            self.load_diff_img=True

        self.load_skin=ctx['load_skin']

    def __len__(self):
        return self.num_samples

    def getitem(self,idx):
        idx=idx%self.num_samples
        sample = {}
        sample_id = self.sample_list[idx]
        rotation_filename = os.path.join(self.rotation_mat_dir, 'rotation_mat_%08d.npy' %sample_id)
        rotations = np.load(rotation_filename).reshape(-1).astype(np.float64)

        sample['rotations'] = torch.from_numpy(rotations)
        sample['index'] = sample_id
        sample['idx'] = idx

        if not self.pd_only:
            if self.load_offset_img:
                offset_img_filename=join(self.offset_img_dir,'offset_img_{:08d}.npy.gz'.format(sample_id))
                with gzip.open(offset_img_filename,'rb') as f:
                    offset_img = np.load(file=f).astype(np.float64)
                if self.use_patches:
                    offset_img=crop_offset_img(offset_img,self.crop)
                sample['offset_img']=torch.from_numpy(offset_img).permute(2,0,1)

            if self.load_diff_img:
                diff_img_filename=join(self.diff_img_dir,'diff_img_{:08d}.npy.gz'.format(sample_id))
                with gzip.open(diff_img_filename,'rb') as f:
                    diff_img=np.load(file=f).astype(np.float64)
                sample['diff_img']=torch.from_numpy(diff_img).permute(2,0,1)

            if self.load_vt_offset:
                vt_offset_filename = os.path.join(self.vt_offset_dir,'displace_%08d.txt' %sample_id)
                vt_offset=np.loadtxt(vt_offset_filename).astype(np.float64)
                if self.use_patches and not self.is_patch_vt_offset:
                    vt_offset=vt_offset[self.vt_ids_in_crop]
                sample['vt_offset'] = torch.from_numpy(vt_offset)

            if self.load_skin:
                skin=np.load(os.path.join(self.skin_dir,'skin_{:08d}.npy'.format(sample_id)))
                if self.use_patches and not self.is_patch_skin:
                    skin=skin[self.vt_ids_in_crop]
                sample['skin']=torch.from_numpy(skin)


            if self.use_normals:
                vn=np.load(os.path.join(self.vn_dir,'normals_{:08d}.npy'.format(sample_id)))
                if self.use_patches:
                    skin=skin[self.vt_ids_in_crop]
                    vn=vn[self.vt_ids_in_crop]
                sample['vn']=torch.from_numpy(vn).squeeze()


        if self.use_patches:
            sample['crop_mask']=torch.from_numpy(self.crop_mask).unsqueeze_(0)

            if self.use_uvn:
                x,y,w,h,side=self.crop
                uvn_hat=self.load_side_uvn_hats(sample_id,side)[self.vt_ids_in_crop]
                sample['uvn_hat']=torch.from_numpy(uvn_hat)
        else:
            if self.load_masks:
                sample['front_mask'] = torch.from_numpy(self.mask[:, :, 0]).unsqueeze_(0)
                sample['back_mask'] = torch.from_numpy(self.mask[:, :, 1]).unsqueeze_(0)

            if self.use_uvn:
                front_uvn_hat=self.load_side_uvn_hats(sample_id,'front')
                back_uvn_hat=self.load_side_uvn_hats(sample_id,'back')
                sample['front_uvn_hat']=torch.from_numpy(front_uvn_hat)
                sample['back_uvn_hat']=torch.from_numpy(back_uvn_hat)

            if self.load_skin_imgs:
                skin_img_path=join(self.skin_img_dir,'skin_img_{:08d}.npy.gz'.format(sample_id))
                with gzip.open(skin_img_path,'rb') as f:
                    skin_img=np.load(file=f)
                sample['skin_img']=torch.from_numpy(skin_img).permute(2,0,1)

        return sample

    def __getitem__(self, idx):
        sample=self.getitem(idx)
        if self.mode=='train' and self.use_hero:
            sim_sample=self.sim_dataset.getitem(self.sim_ids[idx%len(self.sim_ids)])
            sample['sim']=sim_sample
        return sample

        # while True:
        #     try:
        #         return self.getitem(idx)
        #     except Exception as e:
        #         print('dataloader error',e)
        #     new_idx=random.randint(0,self.num_samples-1)
        #     print('idx',idx,'sample_id',self.sample_list[idx],'error,try new idx',new_idx)
        #     idx=new_idx

    def reset_sim_ids(self):
        if not self.use_hero or self.sim_dataset is None:
            assert(False)
        self.sim_ids=torch.randperm(len(self.sim_dataset))

    def load_side_uvn_hats(self,id,side):
        if side=='front':
            front_uhat=np.load(os.path.join(self.uvn_dir,'front_uhats_{:08d}.npy'.format(id)))
            front_uhat=np.expand_dims(front_uhat,2)
            front_vhat=np.load(os.path.join(self.uvn_dir,'front_vhats_{:08d}.npy'.format(id)))
            front_vhat=np.expand_dims(front_vhat,2)
            front_nhat=np.load(os.path.join(self.uvn_dir,'front_nhats_{:08d}.npy'.format(id)))
            front_nhat=np.expand_dims(front_nhat,2)
            front_uvn_hat=np.concatenate([front_uhat,front_vhat,front_nhat],axis=2)
            return front_uvn_hat
        elif side=='back':
            back_uhat=np.load(os.path.join(self.uvn_dir,'back_uhats_{:08d}.npy'.format(id)))
            back_uhat=np.expand_dims(back_uhat,2)
            back_vhat=np.load(os.path.join(self.uvn_dir,'back_vhats_{:08d}.npy'.format(id)))
            back_vhat=np.expand_dims(back_vhat,2)
            back_nhat=np.load(os.path.join(self.uvn_dir,'back_nhats_{:08d}.npy'.format(id)))
            back_nhat=np.expand_dims(back_nhat,2)
            back_uvn_hat=np.concatenate([back_uhat,back_vhat,back_nhat],axis=2)
            return back_uvn_hat
        else:
            print('unseen side',side)
            assert(False)

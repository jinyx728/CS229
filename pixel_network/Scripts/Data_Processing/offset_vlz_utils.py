######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import numpy as np
from PIL import Image
from region_utils import RegionManager

def save_offset_img(out_path,offset,mask=None,maxp=None,minp=None):
    maxp,minp=np.max(offset),np.min(offset)
    img=(offset-minp)/(maxp-minp)
    if mask is not None:
        mask=np.expand_dims(mask,2)
        img=np.concatenate([img,mask],axis=2)
    img=np.uint8(img*255)
    img=np.flip(img,axis=0)
    img=np.flip(img,axis=1)
    Image.fromarray(img.squeeze()).save(out_path)

def save_front_back(samples_dir,test_id,out_dir):
    npy_path=os.path.join(samples_dir,'{:08d}/pred_offset_img.npy'.format(test_id))
    offset_npy=np.load(npy_path)
    img_size=offset_npy.shape[0]

    shared_data_dir='../../shared_data'
    mask_path=os.path.join(shared_data_dir,'offset_img_mask_128.npy')
    mask=np.load(mask_path)
    front_mask=mask[:,:,0]
    back_mask=mask[:,:,1]

    gt_dir='/data/zhenglin/poses_v3/lowres_offset_npys'
    gt_path=os.path.join(gt_dir,'offset_{:08d}.npy'.format(test_id))
    gt_offset=np.load(gt_path)
    maxp,minp=np.max(gt_offset),np.min(gt_offset)

    front_offset=offset_npy[:,:,:3]
    save_offset_img(os.path.join(out_dir,'front.png'),front_offset,mask=front_mask,maxp=maxp,minp=minp)
    back_offset=offset_npy[:,:,3:6]
    save_offset_img(os.path.join(out_dir,'back.png'),back_offset,mask=back_mask,maxp=maxp,minp=minp)

def combine_region_imgs(samples_dir,test_id,img_size=512):
    region_dir=os.path.join(samples_dir,'{:08d}'.format(test_id))
    region_manager=RegionManager()
    front_img=np.zeros((img_size,img_size,3))
    front_mask=np.zeros((img_size,img_size))
    back_img=np.zeros((img_size,img_size,3))
    back_mask=np.zeros((img_size,img_size))
    for region_id in range(region_manager.n_regions):
        crop=region_manager.get_region_crop(region_id)
        x,y,w,h,side=crop
        mask=region_manager.get_region_mask(region_id,crop)
        region_name=region_manager.region_names[region_id]
        region_offset_img_path=os.path.join(region_dir,'{}_offset_img.npy'.format(region_name))
        region_offset_img=np.load(region_offset_img_path)
        print('region_offset_img',region_offset_img.shape,'mask',mask.shape)
        if side=='front':
            front_img[y:y+h,x:x+w,:]+=region_offset_img*np.expand_dims(mask,2)
            front_mask[y:y+h,x:x+w]+=mask
        elif side=='back':
            back_img[y:y+h,x:x+w,:]+=region_offset_img*np.expand_dims(mask,2)
            back_mask[y:y+h,x:x+w]+=mask
        else:
            print('unrecognized side',side)
    # for safe division
    front_mask[front_mask<=0]=1
    back_mask[back_mask<=0]=1
    front_img/=front_mask.reshape((img_size,img_size,1))
    back_img/=back_mask.reshape((img_size,img_size,1))

    shared_data_dir='../../shared_data'
    mask_path=os.path.join(shared_data_dir,'offset_img_mask_512.npy')
    mask=np.load(mask_path)
    # front_mask=mask[:,:,0]
    # back_mask=mask[:,:,1]

    gt_dir='/data/zhenglin/poses_v3/lowres_offset_npys'
    gt_path=os.path.join(gt_dir,'offset_{:08d}.npy'.format(test_id))
    gt_offset=np.load(gt_path)
    maxp,minp=np.max(gt_offset),np.min(gt_offset)

    save_offset_img(os.path.join(region_dir,'front.png'),front_img,mask=mask[:,:,0],maxp=maxp,minp=minp)
    save_offset_img(os.path.join(region_dir,'back.png'),back_img,mask=mask[:,:,1],maxp=maxp,minp=minp)

if __name__=='__main__':
    test_id=2100
    # samples_dir='/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/Learning/rundir/lowres_normal/test/eval_test/'
    # save_front_back(samples_dir,test_id,os.path.join(samples_dir,'{:08d}'.format(test_id)))

    samples_dir='/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/Learning/rundir/lowres_normal_regions/eval_test_test'
    combine_region_imgs(samples_dir,test_id)

######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
def get_color_map():
    color_map=np.zeros((256,3))
    red=np.array([1,0,0])
    blue=np.array([0,0,1])
    white=np.array([1,1,1])
    for i in range(0,128):
        a=i/256*2
        color_map[i]=blue*(1-a)+white*a
    for i in range(128,256):
        a=i/256*2
        color_map[i]=red*(a-1)+white*(2-a)
    return (color_map*255).astype(np.uint8)

def from_offset_img_to_rgb_img(offset_img,mask,flip=True, arr_stats={'minval':np.full(3,-1e-1),'maxval':np.full(3,1e-1)}):
    offset_img=np.concatenate([offset_img, np.zeros((offset_img.shape[0],offset_img.shape[1]))[...,None]], axis=2)
    h, w, c = offset_img.shape
    assert(c==3)
    # zero centered
    # print('size',np.vstack([np.abs(arr_stats['minval']),arr_stats['maxval']]).shape)
    ranges = np.max(np.vstack([np.abs(arr_stats['minval']),arr_stats['maxval']]),axis=0)
    color_map = get_color_map()

    grey_img=np.clip(((offset_img+ranges)/(2*ranges)*255),0,255).astype(np.uint8)
    # if flip:
    #     grey_img=np.flip(grey_img,axis=0)

    rgb_img=color_map[grey_img]

    if mask.ndim==2:
        mask=np.expand_dims(mask,2)
    mask=np.expand_dims(mask,3)
    mask=np.tile(mask,(1,1,3,3))
    rgb_img[mask<=0]=0
    rgb_img=np.transpose(rgb_img,(0,2,1,3)).reshape((h,w*3,3))
    if flip:
        rgb_img=np.flip(rgb_img,axis=0)

    return rgb_img

def concatenate_imgs(imgs_2d_list):
    rows=[]
    for imgs_1d_list in imgs_2d_list:
        rows.append(np.concatenate(imgs_1d_list,axis=1))
    return np.concatenate(rows,axis=0)

def from_offset_img_to_rgb_img_both_sides(offset_img,front_mask,back_mask,arr_stats={'minval':np.full(6,-1e-1),'maxval':np.full(6,1e-1)}):
    front_offset_img=np.transpose(offset_img[:2,:,:],(1,2,0))
    front_arr_stats={name:stats[:3] for name,stats in arr_stats.items()}
    front_rgb_img=from_offset_img_to_rgb_img(front_offset_img,front_mask,arr_stats=front_arr_stats)
    back_offset_img=np.transpose(offset_img[2:4,:,:],(1,2,0))
    back_arr_stats={name:stats[3:] for name,stats in arr_stats.items()}
    back_rgb_img=from_offset_img_to_rgb_img(back_offset_img,back_mask,arr_stats=back_arr_stats)
    return [front_rgb_img,back_rgb_img]

def vlz_pd_offset_img_both_sides(pd_offset_img,gt_offset_img,front_mask,back_mask,arr_stats={'minval':np.full(6,-1e-1),'maxval':np.full(6,1e-1)}):
    pd_rgb_img=from_offset_img_to_rgb_img_both_sides(pd_offset_img,front_mask,back_mask,arr_stats=arr_stats)
    gt_rgb_img=from_offset_img_to_rgb_img_both_sides(gt_offset_img,front_mask,back_mask,arr_stats=arr_stats)
    err_offset_img=pd_offset_img-gt_offset_img
    err_stats={'minval':np.full(6,-1e-2),'maxval':np.full(6,1e-2)}
    err_rgb_img=from_offset_img_to_rgb_img_both_sides(err_offset_img,front_mask,back_mask,arr_stats=err_stats)
    return concatenate_imgs([gt_rgb_img,pd_rgb_img,err_rgb_img])

def vlz_pd_offset_img(pd_offset_img,gt_offset_img,mask,arr_stats={'minval':np.full(3,-1e-1),'maxval':np.full(3,1e-1)}):
    # print('pd_offset_img',pd_offset_img.shape,'mask',mask.shape)
    pd_offset_img=np.transpose(pd_offset_img,(1,2,0))
    # print('pd_offset_img',pd_offset_img.shape)
    pd_rgb_img=from_offset_img_to_rgb_img(pd_offset_img,mask,arr_stats=arr_stats)
    gt_offset_img=np.transpose(gt_offset_img,(1,2,0))
    gt_rgb_img=from_offset_img_to_rgb_img(gt_offset_img,mask,arr_stats=arr_stats)
    err_offset_img=pd_offset_img-gt_offset_img
    err_stats={'minval':np.full(3,-1e-2),'maxval':np.full(3,1e-2)}
    err_rgb_img=from_offset_img_to_rgb_img(err_offset_img,mask,arr_stats=err_stats)
    return np.concatenate([gt_rgb_img,pd_rgb_img,err_rgb_img],axis=0)

######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
import torch
import os
from os.path import join
import gzip

def uvn_to_xyz_img(uvn_offset_img,uvn_hat_img):
    H,W,_=uvn_offset_img.size()
    uvn_hats=uvn_hat_img.view(-1,3,3).permute(0,2,1)
    uvn_offsets=uvn_offset_img.view(-1,3,1)
    # print(torch.max(torch.norm(uvn_hats[:,:,0],dim=1)).item(),torch.max(torch.norm(uvn_hats[:,:,1],dim=1)).item(),torch.max(torch.norm(uvn_hats[:,:,2],dim=1)).item())
    xyz_offsets=uvn_hats.matmul(uvn_offsets)
    xyz_img=xyz_offsets.view(H,W,6)
    return xyz_img

if __name__=='__main__':
    data_root_dir='/data/zhenglin/poses_v3'
    sample_id=106
    uvn_offset_img_path='opt_test/pd_uvn_img_{:08d}.npy'.format(sample_id)
    uvn_offset_img=torch.from_numpy(np.load(uvn_offset_img_path))
    uvn_hat_img_path=join(data_root_dir,'midres_uvnhat_imgs_128/hat_img_{:08d}.npy.gz'.format(sample_id))
    with gzip.open(uvn_hat_img_path,'rb') as f:
    	uvn_hat_img=torch.from_numpy(np.load(file=f)).to(dtype=torch.double)
    	# print(uvn_hat_img.shape)
    print('uvn_offset_img',uvn_offset_img.shape)
    xyz_img=uvn_to_xyz_img(uvn_offset_img,uvn_hat_img)
    xyz_img_path='opt_test/pd_img_{:08d}.npy'.format(sample_id)
    np.save(xyz_img_path,xyz_img.numpy())
######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,isdir
import numpy as np
import gzip

def merge_imgs(uhat_dir,vhat_dir,nhat_dir,out_dir,start,end):
    for sample_id in range(start,end+1):
        print(sample_id)
        uhat=np.load(join(uhat_dir,'hat_img_{:08d}.npy'.format(sample_id)))
        vhat=np.load(join(vhat_dir,'hat_img_{:08d}.npy'.format(sample_id)))
        nhat=np.load(join(nhat_dir,'hat_img_{:08d}.npy'.format(sample_id)))
        # print(np.linalg.norm(uhat[:,:,:3]-vhat[:,:,:3]))
        uvnhat=np.concatenate([uhat[:,:,:3],vhat[:,:,:3],nhat[:,:,:3],uhat[:,:,3:],vhat[:,:,3:],nhat[:,:,3:]],axis=2)
        # print(uvnhat.shape)
        save_path=join(out_dir,'hat_img_{:08d}.npy.gz'.format(sample_id))
        with gzip.open(save_path,'wb') as f:
            np.save(file=f,arr=uvnhat)

if __name__=='__main__':
    data_root_dir='/data/zhenglin/poses_v3'
    start=106
    end=106
    out_dir=join(data_root_dir,'midres_uvnhat_imgs')
    if not isdir(out_dir):
        os.makedirs(out_dir)
    merge_imgs(join(data_root_dir,'midres_uhat_imgs'),join(data_root_dir,'midres_vhat_imgs'),join(data_root_dir,'midres_nhat_imgs'),out_dir,start,end)

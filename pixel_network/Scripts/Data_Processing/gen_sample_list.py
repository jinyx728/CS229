######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,isfile
import numpy as np


def gen_sample_list(mode,total_midres=-1):
    data_root_dir='/data/zhenglin/poses_v3'
    midres_dir=join(data_root_dir,'midres_uvn_offset_imgs')
    lowres_list_path=join(data_root_dir,'sample_lists/lowres_{}_samples.txt'.format(mode))
    lowres_list=np.loadtxt(lowres_list_path).astype(np.int32)
    midres_list_path=join(data_root_dir,'sample_lists/midres_{}_samples.txt'.format(mode))
    midres_list=np.loadtxt(midres_list_path).astype(np.int32)
    midres_set=set(midres_list.tolist())
    # total_midres=4000

    mix_midres_ids=[]
    mix_lowres_ids=[]
    n_midres=0
    for sample_id in lowres_list:
        if sample_id in midres_set:
            midres_path=join(midres_dir,'offset_img_{:08d}.npy'.format(sample_id))
            if isfile(midres_path) and n_midres<total_midres:
                mix_midres_ids.append(sample_id)
                n_midres+=1
            elif not isfile(midres_path):
                print(midres_path,'does not exists!')
            else:
                mix_lowres_ids.append(sample_id)
        else:
            mix_lowres_ids.append(sample_id)

    print('midres',len(mix_midres_ids),'lowres',len(mix_lowres_ids))

    mix_midres_list_path=join(data_root_dir,'sample_lists/mix_midres_{}_samples.txt'.format(mode))
    np.savetxt(mix_midres_list_path,np.array(mix_midres_ids).astype(np.int32),fmt='%d')
    mix_lowres_list_path=join(data_root_dir,'sample_lists/mix_lowres_{}_samples.txt'.format(mode))
    np.savetxt(mix_lowres_list_path,np.array(mix_lowres_ids).astype(np.int32),fmt='%d')


if __name__=='__main__':
    gen_sample_list('train',total_midres=4000)
    gen_sample_list('val',total_midres=10000)
    gen_sample_list('test',total_midres=10000)
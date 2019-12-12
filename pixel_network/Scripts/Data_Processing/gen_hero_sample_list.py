######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,isfile
import numpy as np


def gen_sample_list(mode,total_hero=-1):
    assert(total_hero>0)
    data_root_dir='/data/zhenglin/poses_v3'
    midres_list=np.loadtxt(join(data_root_dir,'sample_lists/mix_midres_{}_samples.txt'.format(mode))).astype(np.int32)
    if total_hero<len(midres_list):
        midres_list=midres_list[:total_hero]
    np.savetxt(join(data_root_dir,'sample_lists/hero_1k_midres_{}_samples.txt'.format(mode)),midres_list,fmt='%d')

    lowres_list=np.loadtxt(join(data_root_dir,'sample_lists/mix_lowres_{}_samples.txt'.format(mode))).astype(np.int32)
    np.savetxt(join(data_root_dir,'sample_lists/hero_1k_lowres_{}_samples.txt'.format(mode)),lowres_list,fmt='%d')



if __name__=='__main__':
    gen_sample_list('train',total_hero=1000)
    gen_sample_list('val',total_hero=100)
    gen_sample_list('test',total_hero=100)
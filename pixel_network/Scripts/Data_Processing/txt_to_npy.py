######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,isfile,isdir
import numpy as np
from pyquaternion import Quaternion

def txt_to_np(in_dir,out_dir):
    if not isdir(out_dir):
        os.makedirs(out_dir)
    files=os.listdir(in_dir)
    for i in range(0,2248):
        in_path=join(in_dir,'rotation_{:08d}.txt'.format(i+1))
        qs=np.loadtxt(in_path)
        ms=[]
        for qi in [1,2,10,11,4,5,6,7,8,9]:
            q=qs[qi]
            ms.append(Quaternion(q).rotation_matrix.reshape(-1))
        out_path=join(out_dir,'rotation_mat_{:08d}.npy'.format(i))
        print('save to',out_path)
        np.save(out_path,np.array(ms))
        # break

if __name__=='__main__':
    txt_to_np('/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/Learning2/Scripts/Data_Processing/joint_test/bld_rots','/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/joint_data/seq1/rotation_matrices')
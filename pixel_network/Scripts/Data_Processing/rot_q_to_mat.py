######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,isdir
import numpy as np
import argparse
from pyquaternion import Quaternion

['LowerBack','Spine','Neck','Neck1','LeftShoulder','LeftArm','LeftForeArm','RightShoulder','RightArm','RightForeArm']
def rot_q_to_mat(in_dir,out_dir,start,end):
    if not isdir(out_dir):
        os.makedirs(out_dir)
    for sample_id in range(start,end+1):
        in_path=join(in_dir,'rotation_{:08d}.txt'.format(sample_id))
        q=np.loadtxt(in_path)
        out_rot=[]
        for i in [1,2,10,11,4,5,6,7,8,9]:
            out_rot.append(Quaternion(q[i]).rotation_matrix.reshape(-1))
        out_path=join(out_dir,'rotation_mat_{:08d}.npy'.format(sample_id))
        print('write to',out_path)
        np.save(out_path,np.array(out_rot))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-in_dir')
    parser.add_argument('-out_dir')
    parser.add_argument('-start',type=int,default=0)
    parser.add_argument('-end',type=int,default=2248)
    args=parser.parse_args()

    rot_q_to_mat(args.in_dir,args.out_dir,args.start,args.end)
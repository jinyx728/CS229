######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,isfile,isdir
import numpy as np
from obj_io import Obj,read_obj
import argparse

def tri_to_npy(tri_dir,npy_dir,tri_prefix,npy_prefix,start,end):
    if not isdir(npy_dir):
        os.makedirs(npy_dir)
    for sample_id in range(start,end+1):
        tri_file=join(tri_dir,'{}_{:08d}.tri.gz'.format(tri_prefix,sample_id))
        obj_file=join(tri_dir,'{}_{:08d}.obj'.format(tri_prefix,sample_id))
        cmd='$PHYSBAM/Tools/tri2obj/tri2obj {} {}'.format(tri_file,obj_file)
        os.system(cmd)
        o=read_obj(obj_file)
        skin_file=join(npy_dir,'{}_{:08d}.npy'.format(npy_prefix,sample_id))
        np.save(skin_file,o.v)
        print('finish',sample_id)


if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('-data_root_dir',default='/data/zhenglin/poses_mocap')
    parser.add_argument('-tri_dir',default='midres_skin_tshirts_13_30')
    parser.add_argument('-tri_prefix',default='skin_tshirt')
    parser.add_argument('-npy_dir',default='midres_skin_npys_13_30')
    parser.add_argument('-npy_prefix',default='skin')
    parser.add_argument('-start',type=int,default=0)
    parser.add_argument('-end',type=int,default=0)
    args=parser.parse_args()

    tri_to_npy(join(args.data_root_dir,args.tri_dir),join(args.data_root_dir,args.npy_dir),args.tri_prefix,args.npy_prefix,args.start,args.end)

######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
import os
from os.path import join,isfile
import argparse


def check_dir(dir_path,shape,id_range):
    def check_id(id):
        path=join(dir_path,'offset_img_{:08d}.npy'.format(id))
        arr=np.load(path)
        if arr.shape!=shape:
            raise Exception(path,'not exists')
    for i in id_range:
        # check_id(i)
        try:
            check_id(i)
        except Exception as e:
            print('failed',i)

        if i%1000==0:
            print('progress',i)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-start',type=int,default=0)
    parser.add_argument('-end',type=int,default=15000)
    parser.add_argument('-res',type=int,choices=[256,512],default=512)
    args=parser.parse_args()

    if args.res==512:
        check_dir('/data/zhenglin/poses_v3/midres_uvn_offset_imgs',(512,512,6),range(args.start,args.end+1))
    elif args.res==256:
        check_dir('/data/zhenglin/poses_v3/midres_uvn_offset_imgs_256',(256,256,6),range(args.start,args.end+1))

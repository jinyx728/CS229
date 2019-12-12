######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import argparse
import numpy as np
from obj_io import Obj,read_obj,write_obj

def merge_obj_dirs(obj_dirs,out_dir,offsets):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    obj_dir0=obj_dirs[0]
    files=os.listdir(obj_dir0)
    counter=0
    for file in files:
        merged_vts=[]
        merged_fcs=[]
        if not file.endswith('.obj'):
            continue
        vertex_offset=0
        n_obj_dirs=len(obj_dirs)
        filename=file[:-4]
        for dir_i in range(n_obj_dirs):
            obj_dir=obj_dirs[dir_i]
            offset=offsets[dir_i]
            obj_path=os.path.join(obj_dir,file)
            # print('obj_path',obj_path)
            if not os.path.isfile(obj_path):
                continue
            obj=read_obj(obj_path)
            merged_vts.append(obj.v+offset)
            merged_fcs.append(obj.f+vertex_offset)
            n_vts=len(obj.v)
            vertex_offset+=n_vts

        if len(merged_vts)==len(obj_dirs):
            # print('len',len(merged_vts))
            merged_obj=Obj(v=np.concatenate(merged_vts,axis=0),f=np.concatenate(merged_fcs,axis=0))
        else:
            continue
        # elif len(merged_vts)==1:
        #     merged_obj=Obj(v=merged_vts[0],f=merged_fcs[0])
        print('merge_obj',file[:-4])
        merged_obj_path=os.path.join(out_dir,file)
        write_obj(merged_obj,merged_obj_path)
        # counter+=1
        # if counter>10:
        #     break


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    # parser.add_argument('-obj_dir',default='out_objs/lowres_normal_regions')
    # parser.add_argument('-tri_dir',default='out_tris/lowres_normal_regions')
    args=parser.parse_args()

    obj_dirs=['../../rundir/hero_pgan/D_hero_R1_100/pd_8','../../rundir/hero_pgan/D_hero_R1_100/gt_8']
    offsets=[np.array([0,0,0]),np.array([0.7,0,0])]
    tri_dir='../../rundir/hero_pgan/D_hero_R1_100/merged/pd_gt_tris'
    merged_obj_dir='../../rundir/hero_pgan/D_hero_R1_100/merged/pd_gt_objs'

    merge_obj_dirs(obj_dirs,merged_obj_dir,offsets)

    if not os.path.isdir(tri_dir):
        os.makedirs(tri_dir)

    files=os.listdir(merged_obj_dir)
    for file in files:
        if file.endswith('.obj'):
            obj_path=os.path.join(merged_obj_dir,file)
            tri_path=os.path.join(tri_dir,'{}.tri'.format(file[:-4]))

            cmd='/data/zhenglin/PhysBAM/Tools/obj2tri/obj2tri {} {}'.format(obj_path,tri_path)
            print(cmd)
            os.system(cmd)
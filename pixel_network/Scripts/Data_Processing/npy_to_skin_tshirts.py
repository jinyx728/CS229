######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,isdir
import numpy as np
from obj_io import Obj,read_obj,write_obj

def convert(npy_dir,skin_tshirts_dir,f=None):
    npy_files=os.listdir(npy_dir)
    for npy_file in npy_files:
        if not npy_file.endswith('.npy'):
            continue
        name=npy_file[:-4]
        obj_path=join(skin_tshirts_dir,'{}.obj'.format(name))
        tri_path=join(skin_tshirts_dir,'{}.tri'.format(name))
        npy_path=join(npy_dir,npy_file)
        v=np.load(npy_path)
        obj=Obj(v=v,f=f)
        write_obj(obj,obj_path)
        cmd='/data/zhenglin/PhysBAM/Tools/obj2tri/obj2tri {} {}'.format(obj_path,tri_path)
        os.system(cmd)
        print('finish',npy_file)
        # break

if __name__=='__main__':
    data_root_dir='/data/zhenglin/poses_v3'
    npy_dir=join(data_root_dir,'midres_skin_npys')
    skin_tshirts_dir=join(data_root_dir,'midres_skin_tshirts')
    if not isdir(skin_tshirts_dir):
        os.makedirs(skin_tshirts_dir)

    shared_data_dir='../../shared_data_midres'
    flat_tshirt_path=join(shared_data_dir,'flat_tshirt.obj')
    f=read_obj(flat_tshirt_path).f

    convert(npy_dir,skin_tshirts_dir,f=f)


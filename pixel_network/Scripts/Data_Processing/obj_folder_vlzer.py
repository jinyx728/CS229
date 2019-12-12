######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../../../../opengl_3d_proto/Proto/Python')
import os
import numpy as np
from obj_io import Obj,read_obj
from PROTO_DEBUG_UTILS import PROTO_DEBUG_UTILS
import argparse

def write_proto_scenes(in_dirs,out_dir,offsets):
    # write out_dir
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    common_dir=os.path.join(out_dir,'common')
    if not os.path.isdir(common_dir):
        os.mkdir(common_dir)

    proto_debug_utils=PROTO_DEBUG_UTILS()
    first_frame=1
    frame=1
    in_dir0=in_dirs[0]
    files=os.listdir(in_dir0)
    for file in files:
        if not file.endswith('obj'):
            continue
        file_name=file[:-4]
        for i in range(len(in_dirs)):
            in_dir=in_dirs[i]
            file_path=os.path.join(in_dir,file)
            if not os.path.isfile(file_path):
                continue
            obj=read_obj(file_path)
            proto_debug_utils.Add_Mesh(obj.v+offsets[i],obj.f,[[1,1,0]])
        frame_dir=os.path.join(out_dir,'{}'.format(frame))
        if not os.path.isdir(frame_dir):
            os.mkdir(frame_dir)
        proto_debug_utils.Write_Output_Files(out_dir,frame)
        print('frame',frame,'file_name',file_name)
        frame+=1
    last_frame=frame
    with open(os.path.join(common_dir,'first_frame'),'w') as f:
        f.write('{}\n'.format(first_frame))
    with open(os.path.join(common_dir,'last_frame'),'w') as f:
        f.write('{}\n'.format(last_frame))

if __name__=='__main__':
    write_proto_scenes(['out_objs/lowres_normal_regions'],'out_objs/vlz',[np.zeros(3)])


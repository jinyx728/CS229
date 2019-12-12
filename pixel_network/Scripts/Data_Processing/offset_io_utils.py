######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import exists,join
import torch
import numpy as np
from obj_io import Obj,read_obj,write_obj

class OffsetIOManager:
    def __init__(self,res_ctx=None,ctx=None):
        self.skin_dir=res_ctx['skin_dir']
        shared_data_dir=res_ctx['shared_data_dir']
        obj_path=join(shared_data_dir,'flat_tshirt.obj')
        obj=read_obj(obj_path)
        self.fcs=obj.f

    def write_cloth_from_offsets(self,offset,sample_id,out_dir,prefix='cloth'):
        skin_path=join(self.skin_dir,'skin_{:08d}.npy'.format(sample_id))
        skin=np.load(skin_path)
        cloth=skin+offset
        obj=Obj(v=cloth,f=self.fcs)
        out_path=join(out_dir,'{}_{:08d}.obj'.format(prefix,sample_id))
        write_obj(obj,out_path)

    def write_cloth(self,vts,path):
        obj=Obj(v=vts,f=self.fcs)
        write_obj(obj,path)

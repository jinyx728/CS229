######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Train_Conv_Mix')
import os
from os.path import join,isfile,isdir
import numpy as np
from obj_io import Obj,read_obj,write_obj
from patch_utils import PatchManager


class NpyToObj:
    def __init__(self):
        self.shared_data_dir='../../shared_data_midres'
        self.patch_id=13
        flat_obj=read_obj(join(self.shared_data_dir,'flat_tshirt.obj'))
        self.f=flat_obj.f
        self.agg_n_vts=len(flat_obj.v)

        if self.patch_id>=0:
            self.patch_manager=PatchManager(shared_data_dir=self.shared_data_dir)
            self.patch_vt_ids=self.patch_manager.load_patch_vt_ids(self.patch_id)
            self.patch_fc_ids=self.patch_manager.get_patch_fc_ids(self.patch_vt_ids,self.f)
            self.f=self.f[self.patch_fc_ids]

    def npy_to_obj(self,in_path,out_path):
        vts=np.load(in_path)
        agg_vts=np.zeros((self.agg_n_vts,3))
        agg_vts[self.patch_vt_ids,:]=vts
        print('write to',out_path)
        write_obj(Obj(v=agg_vts,f=self.f),out_path)

    def txt_to_obj(self,in_path,out_path):
        vts=np.loadtxt(in_path).reshape((-1,3))
        agg_vts=np.zeros((self.agg_n_vts,3))
        agg_vts[self.patch_vt_ids,:]=vts
        print('write to',out_path)
        write_obj(Obj(v=agg_vts,f=self.f),out_path)

    def npy_to_obj_dir(self,d):
        files=os.listdir(d)
        for file in files:
            if not file.endswith('.npy'):
                continue
            in_path=join(d,file)
            out_path=join(d,file[:-4]+'.obj')
            self.npy_to_obj(in_path,out_path)

    def txt_to_obj_dir(self,d):
        files=os.listdir(d)
        for file in files:
            if not file.endswith('.txt'):
                continue
            in_path=join(d,file)
            out_path=join(d,file[:-4]+'.obj')
            self.txt_to_obj(in_path,out_path)

if __name__=='__main__':
    test=NpyToObj()
    test.npy_to_obj_dir('opt_test/cpp_test_midres/cr/p13/debug')
    test.txt_to_obj_dir('opt_test/cpp_test_midres/cr/p13/debug')

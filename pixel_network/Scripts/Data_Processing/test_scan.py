######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,isfile,isdir
from obj_io import read_obj,write_obj,Obj
from ply_io import write_ply
from lap_utils import build_LB
import numpy as np
import pickle
from scipy.sparse import save_npz,load_npz

class ScanTest():
    def __init__(self):
        self.data_dir='scan_test/m1'

    def test_curvature(self):
        obj=read_obj(join(self.data_dir,'avatar.obj'))
        L_path=join(self.data_dir,'L.npz')
        if not isfile(L_path):
            print('compute L...')
            L=build_LB(obj.v,obj.f)
            save_npz(L_path,L)
        else:
            print('load L...')
            L=load_npz(L_path)

        curvs=L.dot(obj.v)
        curv_mags=np.linalg.norm(curvs,axis=1)
        colors=self.cvt_colors(curv_mags)
        out_path=join(self.data_dir,'avatar_curv.ply')
        print('write to',out_path)
        write_ply(out_path,obj.v,obj.f,colors=colors)

    def cvt_colors(self,arr):
        minv,maxv=np.min(arr),np.max(arr)
        print('minv:{}, maxv:{}'.format(minv,maxv))
        t=(arr-minv)/(maxv-minv)
        t=t.reshape((-1,1))
        color0=np.array([[1,1,1]])
        color1=np.array([[1,0,0]])
        colors=color0*(1-t)+color1*t
        colors=np.uint8(colors*255)
        return colors

if __name__=='__main__':
    test=ScanTest()
    test.test_curvature()
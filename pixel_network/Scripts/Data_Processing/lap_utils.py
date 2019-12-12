######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, diags
from obj_io import Obj,read_obj,write_obj
from ply_io import write_ply
import os
from os.path import join,isdir
from timeit import timeit

def get_cot(v0, v1):
    EPISILON = 1e-12
    denom = np.linalg.norm(np.cross(v0, v1))
    if denom < EPISILON:
        print('denom=',denom,'<',EPISILON)
        return 0
    numer = np.inner(v0, v1)
    return numer / denom

@timeit
def build_LB(vertices, faces):
    N, D = vertices.shape
    LB = lil_matrix((N, N))
    for face in faces:
        i0 = face[0]
        i1 = face[1]
        i2 = face[2]
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        cot0 = get_cot(v1-v0, v2-v0)
        cot1 = get_cot(v2-v1, v0-v1)
        cot2 = get_cot(v0-v2, v1-v2)
        LB[i0,i0] += cot1 + cot2
        LB[i0,i1] += -cot2
        LB[i0,i2] += -cot1
        LB[i1,i1] += cot2 + cot0
        LB[i1,i0] += -cot2
        LB[i1,i2] += -cot0
        LB[i2,i2] += cot0 + cot1
        LB[i2,i0] += -cot1
        LB[i2,i1] += -cot0

    return LB.tocsc()

class LapTest:
    def __init__(self):
        self.data_root_dir='/data/zhenglin/poses_v3'
        # self.out_img_dir='opt_test/lap_test'
        # self.shared_data_dir='../../shared_data'
        # self.gt_offset_dir=join(self.data_root_dir,'lowres_offset_npys')
        # self.skin_dir=join(self.data_root_dir,'lowres_skin_npys')
        # self.lap_dir=join(self.data_root_dir,'lowres_lap_npys')

        self.out_img_dir='opt_test/lap_test_midres'
        self.shared_data_dir='../../shared_data_midres'
        self.gt_offset_dir=join(self.data_root_dir,'midres_offset_npys')
        self.skin_dir=join(self.data_root_dir,'midres_skin_npys')
        self.lap_dir=join(self.data_root_dir,'midres_lap_npys')

        tshirt_path=join(self.shared_data_dir,'flat_tshirt.obj')
        tshirt_obj=read_obj(tshirt_path)
        self.L=build_LB(tshirt_obj.v,tshirt_obj.f)
        self.f=tshirt_obj.f

        if not isdir(self.out_img_dir):
            os.makedirs(self.out_img_dir)
        if not isdir(self.lap_dir):
            os.makedirs(self.lap_dir)

    def save_L(self):
        L=self.L
        print('data:',L.data.shape,'indices',L.indices.shape,'indptr',L.indptr.shape)
        np.save(join(self.shared_data_dir,'Lpr.npy'),L.data)
        np.save(join(self.shared_data_dir,'Ljc.npy'),L.indptr)
        np.save(join(self.shared_data_dir,'Lir.npy'),L.indices)

    def test(self,sample_id,write_ply=False):
        gt_offset=np.load(join(self.gt_offset_dir,'offset_{:08d}.npy'.format(sample_id)))
        skin=np.load(join(self.skin_dir,'skin_{:08d}.npy'.format(sample_id)))
        gt_vt=gt_offset+skin
        lap=self.L.dot(gt_vt)
        # lap_path=join(self.lap_dir,'lap_{:08d}.npy'.format(sample_id))
        # np.save(lap_path,lap)
        # print('write',lap_path)
        if write_ply:
            ln=np.linalg.norm(lap,axis=1)
            ln_min,ln_max=np.min(ln),np.max(ln)
            print('ln_min',ln_min,'ln_max',ln_max)
            ln=(ln-ln_min)/(ln_max-ln_min)
            n_vts=gt_vt.shape[0]
            ln=ln.reshape(n_vts,1)
            red=np.tile(np.array([[1,0,0]]),(n_vts,1))
            blue=np.tile(np.array([[0,0,1]]),(n_vts,1))
            colors=red*ln+blue*(1-ln)
            colors=np.uint8(colors*255)
            write_ply(join(self.out_img_dir,'lap_{:08}.ply'.format(sample_id)),gt_vt,self.f,colors=colors)

if __name__=='__main__':
    test=LapTest()
    test.save_L()
    test.test(106)



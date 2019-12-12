######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Train_Conv_Mix')
import numpy as np
import os
from os.path import join,isdir
import cvxpy as cp
from obj_io import Obj,read_obj,write_obj
from timeit import timeit
from vlz_example import VlzExample
from cvxpy_opt import CvxpyOpt
from patch_utils import PatchManager
from cvxpy_opt_func import load_opt_data

class CvxOptTest:
    def __init__(self):
        self.shared_data_dir='../../shared_data'
        self.pd_dir='opt_test/cvxpy'
        self.cr_dir='opt_test/cvxpy'
        self.patch_id=-1
        if self.patch_id>=0:
            self.pd_dir=join(self.pd_dir,'p{:02d}'.format(self.patch_id))
            self.cr_dir=join(self.cr_dir,'p{:02d}'.format(self.patch_id))
        else:
            self.pd_dir=join(self.pd_dir,'whole')
            self.cr_dir=join(self.cr_dir,'whole')
        self.vlz_dir=join(self.cr_dir,'vlz')

        if not isdir(self.pd_dir):
            os.makedirs(self.pd_dir)
        if not isdir(self.cr_dir):
            os.makedirs(self.cr_dir)
        if not isdir(self.vlz_dir):
            os.makedirs(self.vlz_dir)

        res_ctx={'shared_data_dir':self.shared_data_dir}
        ctx={'max_num_constraints':-1}
        m,edges,l0=load_opt_data(res_ctx,ctx)
        self.agg_n_vts=len(m)

        flat_obj=read_obj(join(self.shared_data_dir,'flat_tshirt.obj'))
        self.f=flat_obj.f

        if self.patch_id>=0:
            patch_manager=PatchManager(shared_data_dir=res_ctx['shared_data_dir'])
            self.patch_vt_ids=patch_manager.load_patch_vt_ids(self.patch_id)
            self.patch_fc_ids=patch_manager.get_patch_fc_ids(self.patch_vt_ids,self.f)
            self.patch_edge_ids=patch_manager.get_patch_edge_ids(self.patch_vt_ids,edges)
            m=m[self.patch_vt_ids]
            l0=l0[self.patch_edge_ids]
            edges=patch_manager.get_patch_edges(self.patch_id,edges)

        self.opt=CvxpyOpt(m,edges,l0)

    def read_obj(self,path,patch_id=-1):
        pd_obj=read_obj(path)
        pd_vt=pd_obj.v
        assert(len(pd_vt)==self.agg_n_vts)
        f=pd_obj.f
        if patch_id>=0:
            pd_vt=pd_vt[self.patch_vt_ids]
            f=f[self.patch_fc_ids]
        return pd_vt,f

    def write_obj(self,v,f,out_path,patch_id=-1):
        if patch_id>=0:
            full_v=np.zeros((self.agg_n_vts,3))
            full_v[self.patch_vt_ids,:]=v
            v=full_v
        print('write to',out_path)
        write_obj(Obj(v=v,f=f),out_path)

    def test(self,sample_id):
        v0,f=self.read_obj(join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id)),patch_id=self.patch_id)
        # v0=pd_obj.v
        # for i in range(10):
        v0,_=self.opt.solve(v0)
        # print(dual)
        self.write_obj(v0,f,join(self.cr_dir,'cr_{:08d}.obj'.format(sample_id)),patch_id=self.patch_id)

    def vlz(self,sample_id):
        if not isdir(self.vlz_dir):
            os.makedirs(self.vlz_dir)
        output_directory=join(self.vlz_dir,'Test_{:08d}'.format(sample_id))
        example=VlzExample(output_directory)
        pd_obj=read_obj(join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id)))
        example.draw_edges(pd_obj.v,self.linear_edges,self.linear_rest_lengths)
        example.draw_edges(pd_obj.v,self.bend_edges,self.bend_rest_lengths,compress_color=np.array([1,0,1]),stretch_color=np.array([1,1,0]))
        example.add_frame()
        cr_obj=read_obj(join(self.cr_dir,'cr_{:08d}.obj'.format(sample_id)))
        example.draw_edges(cr_obj.v,self.linear_edges,self.linear_rest_lengths)
        example.draw_edges(cr_obj.v,self.bend_edges,self.bend_rest_lengths,compress_color=np.array([1,0,1]),stretch_color=np.array([1,1,0]))
        example.add_frame()

if __name__=='__main__':
    test=CvxOptTest()
    test.test(16469)
    # opt.vlz(106)

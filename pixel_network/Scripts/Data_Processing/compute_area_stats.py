######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import isfile,isdir,join
import numpy as np
from numpy.linalg import norm
import sys
sys.path.insert(0,'../Train_Conv_Mix')
from obj_io import read_obj,write_obj,Obj
import matplotlib.pyplot as plt
from ply_io import write_ply

class AreaUtils:
    def __init__(self):
        shared_data_dir='../../shared_data'                                                                                   
        flat_tshirt=read_obj(join(shared_data_dir,'flat_tshirt.obj'))
        self.fcs=flat_tshirt.f
        linear_edges=np.loadtxt(join(shared_data_dir,'linear_edges.txt'))
        linear_l0=np.loadtxt(join(shared_data_dir,'mat_or_med_linear.txt'))
        rest_fc_areas=self.compute_areas_from_edge_lengths(self.fcs,linear_edges,linear_l0)
        self.rest_vt_areas=self.compute_vt_areas(flat_tshirt.v,rest_fc_areas)

    def compute_areas_from_edge_lengths(self,fcs,edges,l0):
        n_vts=np.max(fcs)
        def hash_edge(i0,i1):
            if i0<i1:
                return i0*n_vts+i1
            else:
                return i1*n_vts+i0
        edge_hash_to_length={}
        for edge,l0i in zip(edges,l0):
            i0,i1=edge
            edge_hash_to_length[hash_edge(i0,i1)]=l0i
        fc_areas=[]
        for fc in fcs:
            i0,i1,i2=fc
            a=edge_hash_to_length[hash_edge(i0,i1)]
            b=edge_hash_to_length[hash_edge(i1,i2)]
            c=edge_hash_to_length[hash_edge(i2,i0)]
            s=(a+b+c)/2
            fc_areas.append(np.sqrt(s*(s-a)*(s-b)*(s-c)))
        return np.array(fc_areas)

    def compute_fc_areas(self,v):
        v0=v[self.fcs[:,0]]
        v1=v[self.fcs[:,1]]
        v2=v[self.fcs[:,2]]
        return norm(np.cross(v1-v0,v2-v0),axis=1)/2

    def compute_vt_areas(self,v,fc_areas=None):
        if fc_areas is None:
            fc_areas=self.compute_fc_areas(v)
        vt_areas=np.zeros(len(v))
        for fc_i in range(len(fc_areas)):
            fc_area=fc_areas[fc_i]
            i0,i1,i2=self.fcs[fc_i]
            vt_areas[i0]+=fc_area
            vt_areas[i1]+=fc_area
            vt_areas[i2]+=fc_area
        return vt_areas

    def compute_vt_area_ratios(self,v):
        return self.compute_vt_areas(v)/self.rest_vt_areas

def write_area_ratio_ply(samples_dir,in_pattern,out_pattern,ratio_range=0.5):
    sample_dirs=os.listdir(samples_dir)
    area_utils=AreaUtils()
    for sample_dir in sample_dirs:
        sample_id=int(sample_dir)
        obj_path=join(samples_dir,sample_dir,in_pattern.format(sample_id))
        obj=read_obj(obj_path)
        v=obj.v
        r=area_utils.compute_vt_area_ratios(v)
        # print('min:',np.min(r),'max:',np.max(r))
        t=((r-1)/ratio_range).clip(-1,1)
        color0=np.array([[1,1,1]])
        color_neg=np.array([[0,0,1]])
        color_pos=np.array([[1,0,0]])
        colors=np.zeros_like(v)
        # print('colors',colors.shape,'t',t.shape)
        pos_i=t>=0
        neg_i=t<0
        pos_t=(t[pos_i]).reshape((-1,1))
        neg_t=(-t[neg_i]).reshape((-1,1))
        colors[pos_i]=(1-pos_t)*color0+pos_t*color_pos
        colors[neg_i]=(1-neg_t)*color0+neg_t*color_neg
        colors=np.uint8(colors*255)
        out_path=join(samples_dir,sample_dir,out_pattern.format(sample_id))
        print('write to',out_path)
        write_ply(out_path,v,obj.f,colors)

if __name__=='__main__':
    write_area_ratio_ply('../../rundir/lowres_vt/uvn/eval_test','cr_ineq_cloth_{:08d}.obj','area_ratio_cr_ineq_{:08d}.ply')

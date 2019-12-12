######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Train_Conv_Mix')
import numpy as np
import os
from os.path import join,isfile
from topo_utils import get_linear_edges,get_bend_edges
from obj_io import Obj,read_obj
class EdgeStatsTool:
    def __init__(self):
        self.res='lowres'
        self.shared_data_dir='../../shared_data_{}'.format(self.res)
        self.data_root_dir='/data/zhenglin/poses_v3'
        self.skin_dir=join(self.data_root_dir,'{}_skin_npys'.format(self.res))
        self.offset_dir=join(self.data_root_dir,'{}_offset_npys'.format(self.res))
        self.sample_list_path=join(self.data_root_dir,'sample_lists/{}_train_samples.txt'.format(self.res))
        obj_path=join(self.shared_data_dir,'flat_tshirt.obj')
        self.obj=read_obj(obj_path)
        self.linear_edges=self.load_linear_edges(self.obj.f)
        self.bend_edges=self.load_bend_edges(self.obj.f)

    def load_linear_edges(self,fcs):
        linear_edge_path=join(self.shared_data_dir,'linear_edges.txt')
        if not isfile(linear_edge_path):
            print(linear_edge_path,'does not exist, compute')
            linear_edges=get_linear_edges(fcs)
            np.savetxt(linear_edge_path,linear_edges)
        else:
            linear_edges=np.loadtxt(linear_edge_path).astype(int)
        return linear_edges

    def load_bend_edges(self,fcs):
        bend_edge_path=join(self.shared_data_dir,'bend_edges.txt')
        if not isfile(bend_edge_path):
            print(bend_edge_path,'does not exist, compute')
            bend_edges=get_bend_edges(fcs)
            np.savetxt(bend_edge_path,bend_edges)
        else:
            bend_edges=np.loadtxt(bend_edge_path).astype(int)
        return bend_edges

    def compute_edge_lengths(self,vt,edges):
        return np.linalg.norm(vt[edges[:,0]]-vt[edges[:,1]],axis=1)

    def compute_med_lengths(self,sample_list,edges):
        agg_edge_lengths=[]
        for sample_id in sample_list:
            skin_path=join(self.skin_dir,'skin_{:08d}.npy'.format(sample_id))
            skin=np.load(skin_path)
            offset_path=join(self.offset_dir,'offset_{:08d}.npy'.format(sample_id))
            offset=np.load(offset_path)
            vt=skin+offset
            agg_edge_lengths.append(self.compute_edge_lengths(vt,edges))
        agg_edge_lengths=np.transpose(np.array(agg_edge_lengths),(1,0))
        print(agg_edge_lengths.shape)
        med_edge_lengths=np.median(agg_edge_lengths,axis=1)
        return med_edge_lengths

    def save_linear_med_lengths(self):
        sample_list=np.loadtxt(self.sample_list_path).astype(int)[:1000]
        med_edge_lengths=self.compute_med_lengths(sample_list,self.linear_edges)
        med_linear_path=join(self.shared_data_dir,'med_linear.txt')
        np.savetxt(med_linear_path,med_edge_lengths)

    def save_bend_med_lengths(self):
        sample_list=np.loadtxt(self.sample_list_path).astype(int)[:1000]
        med_edge_lengths=self.compute_med_lengths(sample_list,self.bend_edges)
        med_bend_path=join(self.shared_data_dir,'med_bend.txt')
        np.savetxt(med_bend_path,med_edge_lengths)

    def process_linear_lengths(self):
        mat_lengths=self.compute_edge_lengths(self.obj.v,self.linear_edges)
        med_lengths=np.loadtxt(join(self.shared_data_dir,'med_linear.txt'))
        lengths=np.array([mat_lengths,med_lengths])
        lengths=np.max(lengths,axis=0)
        save_path=join(self.shared_data_dir,'mat_or_med_linear.txt')
        np.savetxt(save_path,lengths)

    def process_bend_lengths(self):
        mat_lengths=self.compute_edge_lengths(self.obj.v,self.bend_edges)
        med_lengths=np.loadtxt(join(self.shared_data_dir,'med_bend.txt'))
        lengths=np.array([mat_lengths,med_lengths])
        lengths=np.max(lengths,axis=0)
        save_path=join(self.shared_data_dir,'mat_or_med_bend.txt')
        np.savetxt(save_path,lengths)


if __name__=='__main__':
    t=EdgeStatsTool()
    # compute edge lengths
    t.save_linear_med_lengths()
    t.save_bend_med_lengths()
    t.process_linear_lengths()
    t.process_bend_lengths()
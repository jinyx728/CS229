######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Train_Conv_Mix')
import os
from os.path import join,isfile,isdir
from obj_io import read_obj,Obj
import numpy as np
from patch_utils import PatchManager
import matplotlib.pyplot as plt
def compute_edge_lengths(samples_dir,pattern,edges,l0):
    sample_dirs=os.listdir(samples_dir)
    sample_dirs.sort()
    ratios=[]
    for sample_dir in sample_dirs:
        sample_id=int(sample_dir)
        path=join(samples_dir,sample_dir,pattern.format(sample_id))
        obj=read_obj(path)
        v=obj.v
        l=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
        ratio=l/l0
        ratios+=ratio.tolist()
    return ratios

def plot_ratios(path,ratios,title,xlim):
    ax=plt.gca()
    fig=plt.gcf()
    ax.hist(ratios,bins=20,range=xlim)
    fig.savefig(path)

if __name__=='__main__':
    shared_data_dir='../../shared_data_midres/'
    l0=np.loadtxt(join(shared_data_dir,'mat_or_med_linear.txt'))
    edges=np.loadtxt(join(shared_data_dir,'linear_edges.txt')).astype(np.int)

    patch_id=13
    if patch_id>=0:
        patch_manager=PatchManager(shared_data_dir=shared_data_dir)
        patch_vt_ids=patch_manager.load_patch_vt_ids(patch_id)
        patch_edge_ids=patch_manager.get_patch_edge_ids(patch_vt_ids,edges)
        l0=l0[patch_edge_ids]
        edges=patch_manager.get_patch_edges(patch_id,edges)

    ratios_path='figs/0528/ratios_vt.npy'
    png_path='figs/0528/ratios_vt.png'
    # ratios=compute_edge_lengths('../../rundir/midres_vt_patch/t0/eval_test/waist_front_hip_front_hip_right_waist_right','pd_{:08d}.obj',edges,l0)
    # np.save(ratios_path,ratios)
    # ratios=np.load(ratios_path)
    # plot_ratios(png_path,ratios,'w/o cvx',(0.8,1.2))

    ratios=np.load('figs/0528/ratios_avg.npy')
    plot_ratios('figs/0528/ratios_avg.png',ratios,'avg',(0.8,1.2))

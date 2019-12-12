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
# import torch
from cudaqs_func import load_spring_data,load_axial_data
from obj_io import read_obj,write_obj,Obj
import matplotlib.pyplot as plt
from ply_io import write_ply

class EnergyUitls:
    def __init__(self):
        shared_data_dir='../../shared_data'
        self.spring_data=load_spring_data(shared_data_dir)
        self.axial_data=load_axial_data(shared_data_dir)

    def compute_spring_energy(self,v):
        edges,l0,k=self.spring_data
        l=np.linalg.norm(v[edges[:,1]]-v[edges[:,0]],axis=1)
        return np.sum(k*(l-l0)**2)/2

    def compute_axial_energy(self,v):
        i,w,k=self.axial_data
        d=v[i[:,0]]*w[:,:1]+v[i[:,1]]*w[:,1:2]-v[i[:,2]]*w[:,2:3]-v[i[:,3]]*w[:,3:4]
        return np.sum(np.sum(d**2,axis=1)*k)/2

    def compute_energy(self,v):
        return self.compute_spring_energy(v)+self.compute_axial_energy(v)

    def compute_item_energy(self,v):
        edges,l0,k=self.spring_data
        l=np.linalg.norm(v[edges[:,1]]-v[edges[:,0]],axis=1)
        spring_energy=k*(l-l0)**2/2
        compress_energy=np.sum(spring_energy[l<l0])
        stretch_energy=np.sum(spring_energy[l>l0])
        total_energy=compress_energy+stretch_energy+self.compute_axial_energy(v)
        return compress_energy,stretch_energy,total_energy

def compute_energy_stats(samples_dir,gt_pattern,pd_pattern):
    energy_utils=EnergyUitls()
    sample_dirs=os.listdir(samples_dir)
    energy_list=[]
    for sample_dir in sample_dirs:
        sample_id=int(sample_dir)
        gt_path=join(samples_dir,sample_dir,gt_pattern.format(sample_id))
        gt_v=read_obj(gt_path).v
        pd_path=join(samples_dir,sample_dir,pd_pattern.format(sample_id))
        pd_v=read_obj(pd_path).v
        energy_diff=np.abs(energy_utils.compute_energy(pd_v)-energy_utils.compute_energy(gt_v))
        energy_list.append(energy_diff)
    energy_list=np.array(energy_list)
    print('mean:',np.mean(energy_list),'max',np.max(energy_list),'min',np.min(energy_list),'med',np.median(energy_list),'#',len(energy_list))

def compute_item_energy_stats(samples_dir,pd_pattern):
    energy_utils=EnergyUitls()
    sample_dirs=os.listdir(samples_dir)
    compress_energy_list,stretch_energy_list,total_energy_list=[],[],[]
    for sample_dir in sample_dirs:
    # for sample_dir in ['00016469']:
        sample_id=int(sample_dir)
        # gt_path=join(samples_dir,sample_dir,gt_pattern.format(sample_id))
        # gt_v=read_obj(gt_path).v
        # gt_compress_energy,gt_stretch_energy,gt_total_energy=energy_utils.compute_item_energy(gt_v)
        pd_path=join(samples_dir,sample_dir,pd_pattern.format(sample_id))
        pd_v=read_obj(pd_path).v
        pd_compress_energy,pd_stretch_energy,pd_total_energy=energy_utils.compute_item_energy(pd_v)
        compress_energy_list.append(pd_compress_energy)
        stretch_energy_list.append(pd_stretch_energy)
        total_energy_list.append(pd_total_energy)
    compress_energy_list,stretch_energy_list,total_energy_list=np.array(compress_energy_list),np.array(stretch_energy_list),np.array(total_energy_list)
    print('compress',np.mean(compress_energy_list),'stretch',np.mean(stretch_energy_list),'total',np.mean(total_energy_list),'#',len(total_energy_list))

def compute_item_energy_diff_stats(samples_dir,pd_pattern,gt_pattern):
    energy_utils=EnergyUitls()
    sample_dirs=os.listdir(samples_dir)
    compress_energy_list,stretch_energy_list,total_energy_list=[],[],[]
    for sample_dir in sample_dirs:
    # for sample_dir in ['00016469']:
        sample_id=int(sample_dir)
        gt_path=join(samples_dir,sample_dir,gt_pattern.format(sample_id))
        gt_v=read_obj(gt_path).v
        gt_compress_energy,gt_stretch_energy,gt_total_energy=energy_utils.compute_item_energy(gt_v)
        pd_path=join(samples_dir,sample_dir,pd_pattern.format(sample_id))
        pd_v=read_obj(pd_path).v
        pd_compress_energy,pd_stretch_energy,pd_total_energy=energy_utils.compute_item_energy(pd_v)
        compress_energy_diff=np.abs(gt_compress_energy-pd_compress_energy)
        compress_energy_list.append(compress_energy_diff)
        stretch_energy_diff=np.abs(gt_stretch_energy-pd_stretch_energy)
        stretch_energy_list.append(stretch_energy_diff)
        total_energy_list.append(compress_energy_diff+stretch_energy_diff)

    compress_energy_list,stretch_energy_list,total_energy_list=np.array(compress_energy_list),np.array(stretch_energy_list),np.array(total_energy_list)
    print('compress',np.mean(compress_energy_list),'stretch',np.mean(stretch_energy_list),'total',np.mean(total_energy_list),'#',len(total_energy_list))

def plot_energy():
    plot_dir='figs/0812'
    fig_path=join(plot_dir,'test_energy.png')
    # fig_path=join(plot_dir,'train_energy.png')
    fig=plt.gcf()
    ax=plt.gca()
    bars=["Jenny's","Jenny's+vt","Jenny's+vt+physics",'pre_physics','post_physics']
    ax.bar(bars,[0.12831824436543346,0.03602769571677183,0.008226849095943093,0.04900806083924562,0.002779938123506704])
    # ax.bar(bars,[0.13072915899186716,0.03332437483095766,0.008187269677296303,0.046682442408529756,0.003017166643931206])
    plt.xticks(range(len(bars)), bars, rotation=15)
    ax.set_ylim([0,0.145])
    ax.set_title('Test Energy')
    # ax.set_title('Train Energy (1000)')
    plt.savefig(fig_path)

def plot_item_energy():
    plot_dir='figs/0817'
    fig_path=join(plot_dir,'test_energy.png')
    fig=plt.gcf()
    ax=plt.gca()
    bars=["Jenny's","Jenny's+vt","Jenny's+vt+physics",'pre_physics','post_physics','gt']
    index=np.arange(len(bars))
    bar_width=0.35
    y1=[0.10190138515791278,0.041418693710328906,0.004451217708971002,0.04176686648845303,0.0044298621675275185,0.005190824958609337]
    y2=[0.03579679437274743,0.008613439961099251,0.000665990010806172,0.0164675367213621,0.0017865880961612309,0.002670214152837219]
    ax.bar(index+0.15,y1,bar_width,color='b',label='compress')
    ax.bar(index+bar_width+0.15,y2,bar_width,color='r',label='stretch')
    plt.xticks(index+bar_width, bars, rotation=15)
    ax.legend()
    ax.set_ylim([0,0.102])                                                                                                                                                              
    ax.set_title('Test Energy')
    fig.tight_layout()
    # ax.set_title('Train Energy (1000)')
    plt.savefig(fig_path)

def plot_sec3_energy():
    plot_dir='/data/zhenglin/Documents/JCP2019/figs/sec3'
    fig_path=join(plot_dir,'energy_testset.png')
    fig=plt.gcf()
    ax=plt.gca()
    bars=["before",'after']
    index=np.arange(len(bars))
    bar_width=0.35
    # 16469
    # y1=[0.03682535730773449,0.02605200453719693]
    # y2=[0.009990872851506075,0.0007575068216652829]
    # testset
    y1=[0.041418693710328906,0.030731343538567032]
    y2=[0.008613439961099251,0.0007688575166568654]
    ax.bar(index+0.18,y1,bar_width,color='b',label='compression')
    ax.bar(index+bar_width+0.18,y2,bar_width,color='r',label='stretching')
    plt.xticks(index+bar_width, bars)
    ax.legend()
    ax.set_ylim([0,0.0415])                                                                                                                                                              
    ax.set_title('Energy')
    fig.tight_layout()
    # ax.set_title('Train Energy (1000)')
    plt.savefig(fig_path)

def plot_sec4_energy():
    plot_dir='/data/zhenglin/Documents/JCP2019/figs/sec4'
    fig_path=join(plot_dir,'energy_testset.png')
    fig=plt.gcf()
    ax=plt.gca()
    bars=["[30]",'untrained postprocess','trained postprocess']
    index=np.arange(len(bars))
    bar_width=0.35
    # 16469
    # y1=[0.03682535730773449,0.02605200453719693]
    # y2=[0.009990872851506075,0.0007575068216652829]
    # testset
    y1=[0.02409346608619823,0.015804885071005095,0.01635071555068357]
    y2=[0.011261415734997702,0.00013651987774311217,0.00012707974788683763]
    ax.bar(index+0.18,y1,bar_width,color='b',label='compression')
    ax.bar(index+bar_width+0.18,y2,bar_width,color='r',label='stretching')
    plt.xticks(index+bar_width, bars)
    ax.legend()
    ax.set_ylim([0,0.025])                                                                                                                                                              
    ax.set_title('Energy')
    fig.tight_layout()
    # ax.set_title('Train Energy (1000)')
    plt.savefig(fig_path)

if __name__=='__main__':
    # plot_sec4_energy()
    # compress 0.02409346608619823 stretch 0.011261415734997702 total 0.035354881821195935 # 996
    # compress 0.015804885071005095 stretch 0.00013651987774311217 total 0.015941404948748208 # 996
    # compress 0.01635071555068357 stretch 0.00012707974788683763 total 0.016477795298570408 # 996
    # compute_item_energy_diff_stats('../../rundir/lowres_vt/uvn_1e-2/eval_test','pd_cloth_{:08d}.obj','gt_cloth_{:08d}.obj')
    # compute_item_energy_diff_stats('../../rundir/lowres_vt/uvn_1e-2/eval_test','cr_ineq_cloth_{:08d}.obj','gt_cloth_{:08d}.obj')
    # compute_item_energy_diff_stats('../../rundir/lowres_ecos/uvn_1e-2/eval_test','cr_cloth_{:08d}.obj','gt_cloth_{:08d}.obj')
    # compress 0.005190824958609337 stretch 0.002670214152837219 total 0.012625067062878092 # 996
    # compress 0.10190138515791278 stretch 0.03579679437274743 total 0.14094331142831157 # 996
    # compress 0.041418693710328906 stretch 0.008613439961099251 total 0.05311208559744167 # 996
    # compress 0.004451217708971002 stretch 0.000665990010806172 total 0.008857540784726746 # 996
    # compress 0.04176686648845303 stretch 0.0164675367213621 total 0.0616331279021237 # 996
    # compress 0.0044298621675275185 stretch 0.0017865880961612309 total 0.009881057289318667 # 996
    # compute_item_energy_stats('../../rundir/lowres/uvn/eval_test','gt_cloth_{:08d}.obj')
    # compute_item_energy_stats('../../rundir/lowres/uvn/eval_test','pd_cloth_{:08d}.obj')
    # compute_item_energy_stats('../../rundir/lowres_vt/uvn/eval_test','pd_cloth_{:08d}.obj')
    # compute_item_energy_stats('../../rundir/lowres_vt/uvn/eval_test','cr_cloth_{:08d}.obj')
    # compute_item_energy_stats('../../rundir/lowres_cudaqs/uvn/eval_test','pd_cloth_{:08d}.obj')
    # compute_item_energy_stats('../../rundir/lowres_cudaqs/uvn/eval_test','cr_cloth_{:08d}.obj')
    # plot_item_energy()

    # compress 0.030731343538567032 stretch 0.0007688575166568654 total 0.034608496614169856 # 996
    # compute_item_energy_stats('../../rundir/lowres_vt/uvn/eval_test','cr_ineq_cloth_{:08d}.obj')
    # plot_sec3_energy()

    # mean: 0.012625067062878092 max 0.032658013668147585 min 0.00653734277283418 med 0.011919339301126644 # 996
    # mean: 0.14094331142831157 max 0.529049484127888 min 0.07056375086843064 med 0.13179201031208487 # 996
    # mean: 0.061633127902123705 max 0.23141087362611606 min 0.027865291296810916 med 0.05717891441885123 # 996
    # mean: 0.009881057289318667 max 0.021040235636253513 min 0.005070799778896015 med 0.009354422923269828 # 996
    # compute_energy_stats('../../rundir/lowres/uvn/eval_test','gt_cloth_{:08d}.obj')
    # compute_energy_stats('../../rundir/lowres/uvn/eval_test','pd_cloth_{:08d}.obj')
    # compute_energy_stats('../../rundir/lowres_cudaqs/uvn/eval_test','pd_cloth_{:08d}.obj')
    # compute_energy_stats('../../rundir/lowres_cudaqs/uvn/eval_test','cr_cloth_{:08d}.obj')
    # plot_energy()

    # mean: 0.012453334695486474 max 0.03783076709072447 min 0.005958785437249611 med 0.011558873568481037 # 1000
    # mean: 0.14318249368735367 max 0.6403757161990808 min 0.06649566396973888 med 0.1328462066770688 # 1000
    # mean: 0.05913577710401623 max 0.239640689100169 min 0.024659763661843904 med 0.05540198697017622 # 1000
    # mean: 0.009443864983724903 max 0.02169910545512956 min 0.004935053177954106 med 0.00888565902713585 # 1000
    # compute_energy_stats('../../rundir/lowres/uvn/eval_train','gt_cloth_{:08d}.obj')
    # compute_energy_stats('../../rundir/lowres/uvn/eval_train','pd_cloth_{:08d}.obj')
    # compute_energy_stats('../../rundir/lowres_cudaqs/uvn/eval_train','pd_cloth_{:08d}.obj')
    # compute_energy_stats('../../rundir/lowres_cudaqs/uvn/eval_train','cr_cloth_{:08d}.obj')
    # plot_energy()

    # write_area_ratio_ply('../../rundir/lowres/uvn/eval_test','pd_cloth_{:08d}.obj','area_ratio_{:08d}.ply')
    # write_area_ratio_ply('../../rundir/lowres_cudaqs/uvn/eval_test','pd_cloth_{:08d}.obj','area_ratio_pd_{:08d}.ply')
    # write_area_ratio_ply('../../rundir/lowres_cudaqs/uvn/eval_test','cr_cloth_{:08d}.obj','area_ratio_cr_{:08d}.ply')

    # pd
    # mean: 0.05311208559744167 max 0.18656219560420004 min 0.022188922106033987 med 0.0485291492682617 # 996
    # mean: 0.049888929497106715 max 0.14988753831793714 min 0.019900440275501415 med 0.04624746517482811 # 1000
    # cr
    # mean: 0.008857540784726748 max 0.018886309309745775 min 0.004504520795747197 med 0.00838045449673857 # 996
    # mean: 0.008377284988852757 max 0.017792459068028857 min 0.0043180574922201415 med 0.007909824435561121 # 1000
    # compute_energy_stats('../../rundir/lowres_vt/uvn/eval_test','cr_cloth_{:08d}.obj')
    # compute_energy_stats('../../rundir/lowres_vt/uvn/eval_train','cr_cloth_{:08d}.obj')
    # plot_energy()

    # write_area_ratio_ply('../../rundir/lowres_vt/uvn/eval_test','pd_cloth_{:08d}.obj','area_ratio_pd_{:08d}.ply')
    # write_area_ratio_ply('../../rundir/lowres_vt/uvn/eval_test','cr_cloth_{:08d}.obj','area_ratio_cr_{:08d}.ply')

    # mean: 0.12831824436543346 max 0.515246854349316 min 0.05885664129458246 med 0.11966207338578866 # 996
    # mean: 0.03602769571677183 max 0.13147171337668034 min 0.011308458844876426 med 0.03255791010896121 # 996
    # mean: 0.008226849095943093 max 0.040812246378529496 min 0.0010690010156788133 med 0.006871901972121448 # 996
    # mean: 0.04900806083924562 max 0.1987528599579685 min 0.019391846091056486 med 0.04517517778731309 # 996
    # mean: 0.002779938123506704 max 0.017767829629819196 min 9.756094738308865e-06 med 0.002529230031332594 # 996
    # compute_energy_stats('../../rundir/lowres/uvn/eval_test','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj')
    # compute_energy_stats('../../rundir/lowres_vt/uvn/eval_test','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj')
    # compute_energy_stats('../../rundir/lowres_vt/uvn/eval_test','gt_cloth_{:08d}.obj','cr_cloth_{:08d}.obj')
    # compute_energy_stats('../../rundir/lowres_cudaqs/uvn/eval_test','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj')
    # compute_energy_stats('../../rundir/lowres_cudaqs/uvn/eval_test','gt_cloth_{:08d}.obj','cr_cloth_{:08d}.obj')

    # mean: 0.13072915899186716 max 0.6301879977989278 min 0.054842869336901794 med 0.12102195085976389 # 1000
    # mean: 0.03332437483095766 max 0.12992072983149508 min 0.011324051045588897 med 0.030874824278734427 # 1000
    # mean: 0.008187269677296303 max 0.04209228935626915 min 0.0017858584606894722 med 0.006686763700020652 # 1000
    # mean: 0.046682442408529756 max 0.22342291354931582 min 0.017309569980759446 med 0.04322070851723778 # 1000
    # mean: 0.003017166643931206 max 0.018828249157899015 min 1.0397155491229798e-05 med 0.002691356944002775 # 1000

    # compute_energy_stats('../../rundir/lowres/uvn/eval_train','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj')
    # compute_energy_stats('../../rundir/lowres_vt/uvn/eval_train','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj')
    # compute_energy_stats('../../rundir/lowres_vt/uvn/eval_train','gt_cloth_{:08d}.obj','cr_cloth_{:08d}.obj')
    # compute_energy_stats('../../rundir/lowres_cudaqs/uvn/eval_train','gt_cloth_{:08d}.obj','pd_cloth_{:08d}.obj')
    # compute_energy_stats('../../rundir/lowres_cudaqs/uvn/eval_train','gt_cloth_{:08d}.obj','cr_cloth_{:08d}.obj')

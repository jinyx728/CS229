######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,isfile,isdir
import numpy as np
import pickle as pk
# import PSpincalc as sp
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def to_scipy_quat(q):
    return np.array([q[1],q[2],q[3],q[0]])

def compute_stats(out_path,rot_dir,ids):
    joint_names=['LowerBack','Spine','Neck','Neck1','LeftShoulder','LeftArm','LeftForeArm','RightShoulder','RightArm','RightForeArm']
    modes=['yxz','yzx','yzx','yxz','xzy','xzy','xyz','xzy','xzy','xyz']
    # modes=['zxy','xzy','xzy','zxy','yzx','yzx','zyx','yzx','yzx','zyx']

    # modes=['yxz']*len(joint_names)
    # modes=['zyx']*len(joint_names)
    stats={name:[] for name in joint_names}
    for sample_id in ids:
    # for sample_id in [15587]:
    # for sample_id in [1172]:
        if sample_id>=15000:
            continue
        rot_path=join(rot_dir,'rotation_mat_{:08d}.npy'.format(sample_id))
        rot=np.load(rot_path)
        for joint_i in range(len(rot)):
            joint_rot=rot[joint_i].reshape((3,3))
            # angles=sp.DCM2EA(joint_rot,'yxz')[0]
            mode=modes[joint_i].upper()
            # mode='zyx'
            # mode='xyz'
            # angles=sp.DCM2EA(joint_rot,mode)[0]
            angles=R.from_quat(to_scipy_quat(Quaternion(matrix=joint_rot).q)).as_euler(mode)
            # angles=R.from_dcm(joint_rot).as_euler(mode)
            # angles=R.as_euler(mode,)
            # angles=sp.Q2EA(Quaternion(matrix=joint_rot).q,mode)[0]
            sorted_angles=[0,0,0]
            for i in range(3):
                # m=mode[2-i]
                m=mode[i]

                if m=='x' or m=='X': 
                    sorted_angles[1]=angles[i]
                if m=='y' or m=='Y': 
                    sorted_angles[0]=angles[i]
                if m=='z' or m=='Z': 
                    sorted_angles[2]=angles[i]

            joint_name=joint_names[joint_i]

            # if joint_name=='LeftForeArm':
            #     if sorted_angles[2]>1e-2 and sample_id<249:
            #         print('sample_id',sample_id)
            #         print('rot\n',joint_rot)
            #         print('angles',sorted_angles)
            #         exit(0)

            stats[joint_name].append(sorted_angles)
            # if joint_name=='LeftForeArm' or joint_name=='RightForeArm':
                # print(sample_id,'sorted_angles',sorted_angles,angles)

    print('write to',out_path)
    pk.dump(stats,open(out_path,'wb'))

def plot_distr(stats_path,out_dir,bad_stats_path=None):
    if not isdir(out_dir):
        os.makedirs(out_dir)
    stats=pk.load(open(stats_path,'rb'))
    plot_bad_stats=bad_stats_path is not None
    if plot_bad_stats:
        bad_stats=pk.load(open(bad_stats_path,'rb'))

    for k,angles in stats.items():
        angles=np.array(angles)

        fig=plt.gcf()
        ax=plt.gca()
        rangex=np.max(np.abs(angles[:,1]))
        rangez=np.max(np.abs(angles[:,2]))
        if plot_bad_stats:
            bad_angles=np.array(bad_stats[k])
            ax.scatter(bad_angles[:,1],bad_angles[:,2],marker='.',c='r',s=1)
            rangex=max(rangex,np.max(np.abs(bad_angles[:,1])))
            rangez=max(rangez,np.max(np.abs(bad_angles[:,2])))
        ax.scatter(angles[:,1],angles[:,2],marker='.',c='g',s=1)

        rangex=max(rangex,0.1)
        rangez=max(rangez,0.1)

        ax.set_title('{}_XZ'.format(k))
        ax.spines['left'].set_color('black')
        ax.yaxis.set_visible(True)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlim([-rangex,rangex])
        ax.set_ylim([-rangez,rangez])

        out_path=join(out_dir,'{}_XZ.png'.format(k))
        print('save to',out_path)
        plt.savefig(out_path)

        fig=plt.figure()
        ax=plt.gca()
        rangey=np.max(np.abs(angles[:,0]))
        if plot_bad_stats:
            ax.scatter(bad_angles[:,0],np.zeros(len(bad_angles)),marker='.',c='r',s=1)
            rangey=max(rangey,np.max(np.abs(bad_angles[:,0])))
        ax.scatter(angles[:,0],np.zeros(len(angles)),marker='.',c='g',s=1)
        rangey=max(rangey,0.1)

        ax.set_title('{}_Y'.format(k))
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_visible(False)
        ax.set_xlim([-rangey,rangey])
        # ax.hist(angles[:,2],bins=20)
        # ax.set_title('{}_Y'.format(k))
        out_path=join(out_dir,'{}_Y.png'.format(k))
        print('save to',out_path)
        plt.savefig(out_path)

        # break

def get_ids(max_id=15000):
    # return np.loadtxt('/data/zhenglin/poses_v3/sample_lists/lowres_train_samples.txt').astype(int)
    ids_good=[]
    ids_good+=np.loadtxt('/data/zhenglin/poses_v3/sample_lists/lowres_train_samples.txt').astype(int).tolist()
    ids_good+=np.loadtxt('/data/zhenglin/poses_v3/sample_lists/lowres_val_samples.txt').astype(int).tolist()
    ids_good+=np.loadtxt('/data/zhenglin/poses_v3/sample_lists/lowres_test_samples.txt').astype(int).tolist()
    
    tmp_ids=[]
    for i in ids_good:
        if i<max_id:
            tmp_ids.append(i)
    ids_good=tmp_ids

    print('len(ids_good)',len(ids_good))
    ids_bad=[]
    set_good=set(ids_good)
    for i in range(max(ids_good)):
        if not i in set_good:
            ids_bad.append(i)
    print('len(ids_bad',len(ids_bad))
    return ids_good,ids_bad

if __name__=='__main__':
    # ids=np.loadtxt('/data/zhenglin/poses_v3/sample_lists/lowres_train_samples.txt').astype(int)
    # compute_stats('joint_test/rot_stats.pk','/data/zhenglin/poses_v3/rotation_matrices',ids)
    # plot_distr('joint_test/rot_stats.pk','joint_test/rot_stats')

    # ids_good,ids_bad=get_ids()
    # compute_stats('joint_test/rot_stats_good.pk','/data/zhenglin/poses_v3/rotation_matrices',ids_good)
    # compute_stats('joint_test/rot_stats_bad.pk','/data/zhenglin/poses_v3/rotation_matrices',ids_bad)
    plot_distr('joint_test/rot_stats_good.pk','joint_test/rot_stats','joint_test/rot_stats_bad.pk')



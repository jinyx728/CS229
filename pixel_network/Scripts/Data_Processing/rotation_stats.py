######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,isfile
import numpy as np
import argparse

def cal_stats(rot_dir,n_samples=30000):
	rot_sum=0
	rot_sqr=0
	for sample_id in range(n_samples):
		rot=np.load(join(rot_dir,'rotation_mat_{:08d}.npy'.format(sample_id)))
		rot_sum+=rot
		rot_sqr+=rot**2
		if sample_id%100==0:
			print(sample_id)
	rot_mean=rot_sum/n_samples
	rot_std=np.sqrt(rot_sqr/n_samples-rot_mean*rot_mean)
	return rot_mean,rot_std

if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('-rot_dir',default='/data/zhenglin/poses_v3/rotation_matrices')
	args=parser.parse_args()

	mean,std=cal_stats(args.rot_dir)
	print('mean',mean)
	print('std',std)
	shared_data_dir='../../shared_data_midres'
	np.save(join(shared_data_dir,'rotation_mean.npy'),mean)
	np.save(join(shared_data_dir,'rotation_std.npy'),std)
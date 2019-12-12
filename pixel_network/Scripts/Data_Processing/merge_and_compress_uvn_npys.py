######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,exists
import gzip
import numpy as np
import argparse

def compress_folder(in_dir,out_dir):
	if not exists(out_dir):
		os.makedirs(out_dir)
	data_root_dir='/data/zhenglin/poses_v3'
	shared_data_dir='../../shared_data_midres'
	front_vt_ids=np.loadtxt(join(shared_data_dir,'front_vertices.txt')).astype(np.int32)
	back_vt_ids=np.loadtxt(join(shared_data_dir,'back_vertices.txt')).astype(np.int32)
	front_vt_id_set=set(front_vt_ids.tolist())
	back_vt_id_set=set(back_vt_ids.tolist())
	front_intr_vt_ids=[]
	back_intr_vt_ids=[]
	for vt_id in front_vt_ids:
		if not vt_id in back_vt_id_set:
			front_intr_vt_ids.append(vt_id)
	for vt_id in back_vt_ids:
		if not vt_id in front_vt_id_set:
			back_intr_vt_ids.append(vt_id)

	for sample_id in range(15000):
		front_uhat=np.load(join(in_dir,'front_uhats_{:08d}.npy'.format(sample_id)))
		front_vhat=np.load(join(in_dir,'front_vhats_{:08d}.npy'.format(sample_id)))
		front_nhat=np.load(join(in_dir,'front_nhats_{:08d}.npy'.format(sample_id)))
		back_uhat=np.load(join(in_dir,'back_uhats_{:08d}.npy'.format(sample_id)))
		back_vhat=np.load(join(in_dir,'back_vhats_{:08d}.npy'.format(sample_id)))
		back_nhat=np.load(join(in_dir,'back_nhats_{:08d}.npy'.format(sample_id)))
		uvn_hat=np.dstack([front_uhat,front_vhat,front_nhat,back_uhat,back_vhat,back_nhat])
		uvn_hat[back_intr_vt_ids,:,:3]=0
		uvn_hat[front_intr_vt_ids,:,3:]=0
		with gzip.open(join(out_dir,'uvn_hats_{:08d}.npy.gz'.format(sample_id)),'wb') as f:
			np.save(file=f,arr=uvn_hat)
		print(sample_id)

if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('-in_dir')
	parser.add_argument('-out_dir')
	args=parser.parse_args()

	assert(args.in_dir is not None)
	assert(args.out_dir is not None)
	assert(exists(args.in_dir))

	compress_folder(args.in_dir,args.out_dir)
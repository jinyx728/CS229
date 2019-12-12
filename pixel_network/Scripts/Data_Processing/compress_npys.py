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
	files=os.listdir(in_dir)
	files.sort()
	for file in files:
		if not file.endswith('.npy'):
			continue
		arr=np.load(join(in_dir,file))
		with gzip.open(join(out_dir,'{}.gz'.format(file)),'wb') as f:
			np.save(file=f,arr=arr)
		print(file)

if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('-in_dir')
	parser.add_argument('-out_dir')
	args=parser.parse_args()

	assert(args.in_dir is not None)
	assert(args.out_dir is not None)
	assert(exists(args.in_dir))

	compress_folder(args.in_dir,args.out_dir)
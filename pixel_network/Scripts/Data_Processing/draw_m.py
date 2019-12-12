######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
from obj_io import Obj,read_obj
from ply_io import write_ply
import numpy as np
import argparse

def draw_m(obj_path,m_path,ply_path,c0=np.array([1,1,1]),c1=np.array([1,0,0])):
	obj=read_obj(obj_path)
	m=np.load(m_path)
	mmin=np.min(m)
	mmax=np.max(m)
	print('min:',mmin,'max',mmax)
	mmin,mmax=0.5,1.5
	t=(m-mmin)/(mmax-mmin)
	t=np.clip(t,0,1)
	n_vts=t.shape[0]
	t=t.reshape((n_vts,1))
	C0=np.tile(c0.reshape((1,3)),(n_vts,1))
	C1=np.tile(c1.reshape((1,3)),(n_vts,1))
	colors=C0*(1-t)+C1*t
	colors=np.uint8(colors*255)
	print('write to',ply_path)
	write_ply(ply_path,obj.v,obj.f,colors=colors)

if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('-obj_path',default='../../rundir/lowres_ecos/variable_m/eval_test/00001412/pd_cloth_00001412.obj')
	parser.add_argument('-m_path',default='../../rundir/lowres_ecos/variable_m/eval_test/00001412/pd_m_00001412.npy')
	parser.add_argument('-ply_path',default='../../rundir/lowres_ecos/variable_m/eval_test/00001412/pd_mcloth_00001412.ply')
	args=parser.parse_args()

	draw_m(args.obj_path,args.m_path,args.ply_path)
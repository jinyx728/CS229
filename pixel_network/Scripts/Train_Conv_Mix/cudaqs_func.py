######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import os
from os.path import join,isfile,isdir
from patch_utils import PatchManager
import cudaqs

class CudaqsFunction(Function):
	@staticmethod
	def forward(ctx,opts,anchor,stiffen_anchor):
		forward_opt,backward_opt,opt_data,use_variable_m=opts
		x,flags=cudaqs.solve_forward(forward_opt,anchor,stiffen_anchor,opt_data)

		n_samples=len(anchor)
		for sample_i in range(n_samples):
			if flags[sample_i]!=0 and flags[sample_i]!=1:
				print('forward:sample:{},flag:{},set to anchor'.format(sample_i,['SUCCESS','INACCURATE_RESULT','SEARCH_DIR_FAIL','MAX_ITER'][flags[sample_i]]))
				x[sample_i]=anchor[sample_i]

		ctx.backward_opt=backward_opt
		ctx.opt_data=opt_data
		ctx.use_variable_m=use_variable_m
		ctx.save_for_backward(x,anchor,stiffen_anchor)
		return x
	
	@staticmethod
	def backward(ctx,in_grad):
		backward_opt=ctx.backward_opt
		opt_data=ctx.opt_data
		x,anchor,stiffen_anchor=ctx.saved_tensors
		da,dstiffen_anchor,flags=cudaqs.solve_backward(backward_opt,in_grad,x,anchor,stiffen_anchor,opt_data)
		
		n_samples=len(in_grad)
		for sample_i in range(n_samples):
			if flags[sample_i]!=0:
				print('backward:sample:{},flags:{},set to zero'.format(sample_i,['SUCCESS','MAX_CG_ITER'][flags[sample_i]]))
				da[sample_i]=0

		if ctx.use_variable_m:
			return None,da,dstiffen_anchor
		else:
			return None,da,None

cudaqs_func=CudaqsFunction.apply

class CudaqsModule(nn.Module):
	def __init__(self):
		super(CudaqsModule,self).__init__()

	def init(self,spring_data,axial_data,system,forward_opt,backward_opt,opt_data,use_variable_m):
		self.spring_data=spring_data
		self.axial_data=axial_data
		self.system=system
		self.opts=forward_opt,backward_opt,opt_data,use_variable_m

	def forward(self,anchor,stiffen_anchor=None):
		if stiffen_anchor is None:
			batch_size=len(anchor)
			stiffen_anchor=self.stiffen_anchor.repeat(batch_size,1)
		return cudaqs_func(self.opts,anchor,stiffen_anchor)		

def load_anchor_data(shared_data_dir):
	stiffen_anchor_path=join(shared_data_dir,'m.txt')
	stiffen_anchor=np.loadtxt(stiffen_anchor_path)*1e4
	return stiffen_anchor

def load_spring_data(shared_data_dir):
	linear_edges=np.loadtxt(join(shared_data_dir,'linear_edges.txt')).astype(int)
	bend_edges=np.loadtxt(join(shared_data_dir,'bend_edges.txt')).astype(int)
	edges=np.concatenate([linear_edges,bend_edges],axis=0)
	linear_l0=np.loadtxt(join(shared_data_dir,'mat_or_med_linear.txt'))
	bend_l0=np.loadtxt(join(shared_data_dir,'mat_or_med_bend.txt'))
	l0=np.concatenate([linear_l0,bend_l0])

	# m=stiffen_anchor
	# linear_harmonic_m=1/(1/m[linear_edges[:,0]]+1/m[linear_edges[:,1]])
	# print('write',join(shared_data_dir,'linear_harmonic_m.txt'))
	# np.savetxt(join(shared_data_dir,'linear_harmonic_m.txt'),linear_harmonic_m)
	# bend_harmonic_m=1/(1/m[bend_edges[:,0]]+1/m[bend_edges[:,1]])
	# print('write',join(shared_data_dir,'bend_harmonic_m.txt'))
	# np.savetxt(join(shared_data_dir,'bend_harmonic_m.txt'),bend_harmonic_m)

	linear_k=np.loadtxt(join(shared_data_dir,'linear_harmonic_m.txt'))*(10/(1+np.sqrt(2)))
	bend_k=np.loadtxt(join(shared_data_dir,'bend_harmonic_m.txt'))*(2/(1+np.sqrt(2)))
	k=np.concatenate([linear_k,bend_k])
	return edges,l0,k

def load_axial_data(shared_data_dir):
	i_path=join(shared_data_dir,'axial_spring_particles.txt')
	axial_i=np.loadtxt(i_path).astype(np.int)
	w_path=join(shared_data_dir,'axial_particle_weights.txt')
	axial_w=np.loadtxt(w_path).astype(np.float64)

	# m=stiffen_anchor
	# axial_harmonic_m=4/(1/m[axial_i[:,0]]+1/m[axial_i[:,1]]+1/m[axial_i[:,2]]+1/m[axial_i[:,3]])
	# print('save axial_harmonic_m')
	# np.savetxt(join(shared_data_dir,'axial_harmonic_m.txt'),axial_harmonic_m)

	k_path=join(shared_data_dir,'axial_harmonic_m.txt')
	axial_k=np.loadtxt(k_path).astype(np.float64)*1e-1
	return axial_i,axial_w,axial_k

def init_cudaqs_module(res_ctx,ctx):
	print('init_cudaqs_module')

	device,dtype=ctx['device'],torch.double
	batch_size=ctx['batch_size']
	verbose=ctx['verbose'] if 'verbose' in ctx else False
	use_multi_thread=False
	shared_data_dir=res_ctx['shared_data_dir']
	patch_id=ctx['patch_id'] if 'patch_id' in ctx else -1
	stiffen_anchor_factor=ctx['stiffen_anchor_factor']
	use_variable_m=ctx['use_variable_m']

	stiffen_anchor=load_anchor_data(shared_data_dir)	
	edges,l0,k=load_spring_data(shared_data_dir)
	axial_i,axial_w,axial_k=load_axial_data(shared_data_dir)

	if patch_id>=0:
		patch_manager=PatchManager(shared_data_dir=res_ctx['shared_data_dir'])
		patch_vt_ids=patch_manager.load_patch_vt_ids(patch_id)
		patch_edge_ids=patch_manager.get_patch_edge_ids(patch_vt_ids,edges)
		stiffen_anchor=stiffen_anchor[patch_vt_ids]
		edges=patch_manager.get_patch_edges(patch_id,edges)
		l0=l0[patch_edge_ids]
		k=k[patch_edge_ids]
		patch_axial_edge_ids=patch_manager.get_patch_edge_ids(patch_vt_ids,axial_i)
		axial_i=patch_manager.get_patch_edges(patch_id,axial_i)
		axial_w=axial_w[patch_axial_edge_ids]
		axial_k=axial_k[patch_axial_edge_ids]

	stiffen_anchor=torch.from_numpy(stiffen_anchor).to(device=device,dtype=dtype)*stiffen_anchor_factor
	edges=torch.from_numpy(edges).to(device=device,dtype=torch.int32)
	l0=torch.from_numpy(l0).to(device=device,dtype=dtype)
	k=torch.from_numpy(k).to(device=device,dtype=dtype)
	axial_i=torch.from_numpy(axial_i).to(device=device,dtype=torch.int32)
	axial_w=torch.from_numpy(axial_w).to(device=device,dtype=dtype)
	axial_k=torch.from_numpy(axial_k).to(device=device,dtype=dtype)
	n_vts=len(stiffen_anchor)

	module=CudaqsModule()
	cudaqs.init(n_vts,batch_size,use_multi_thread,verbose)
	spring_data=cudaqs.init_spring(edges,l0,k)
	axial_data=cudaqs.init_axial(axial_i,axial_w,axial_k)
	system=cudaqs.init_system(n_vts,spring_data,axial_data)
	forward_opt=cudaqs.init_forward(system)
	backward_opt=cudaqs.init_backward(system,use_variable_m)
	# print('len(edges):',len(edges))
	opt_data=cudaqs.init_opt_data(batch_size,n_vts,len(edges))
	module.init(spring_data,axial_data,system,forward_opt,backward_opt,opt_data,use_variable_m)
	module.stiffen_anchor=stiffen_anchor.unsqueeze(0)

	print('init_cudaqs_module finish')

	return module


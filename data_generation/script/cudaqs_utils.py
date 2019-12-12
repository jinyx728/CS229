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
import cudaqs

class CudaqsFunction(Function):
	@staticmethod
	def forward(ctx,opts,anchor,stiffen_anchor):
		forward_opt,backward_opt,opt_data,use_variable_m=opts
		x,flags=cudaqs.solve_forward(forward_opt,anchor,stiffen_anchor,opt_data)

		# n_samples=len(anchor)
		# for sample_i in range(n_samples):
		# 	if flags[sample_i]!=0 and flags[sample_i]!=1 and flags[sample_i]!=2:
		# 		print('forward:sample:{},flag:{},set to anchor'.format(sample_i,['SUCCESS','INACCURATE_RESULT','SEARCH_DIR_FAIL','MAX_ITER'][flags[sample_i]]))
		# 		x[sample_i]=anchor[sample_i]

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
		return cudaqs_func(self.opts,anchor,stiffen_anchor.contiguous())		

def load_anchor_data(shared_data_dir,m_init_file=None):
	if m_init_file is not None:
		print('load stiffen_anchor from',m_init_file)
		stiffen_anchor_path=m_init_file
	else:
		stiffen_anchor_path=join(shared_data_dir,'m.txt')

	stiffen_anchor=np.loadtxt(stiffen_anchor_path)*1e4
	return stiffen_anchor

def load_spring_data(shared_data_dir,use_constant_k):
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

	if use_constant_k:
		linear_k=1/linear_l0*(10/(1+np.sqrt(2)))*0.02
		bend_k=1/bend_l0*(2/(1+np.sqrt(2)))*0.02*5
	else:
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
	axial_k=np.loadtxt(k_path).astype(np.float64)*5e0
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
	use_constant_k=ctx['use_constant_k'] if 'use_constant_k' in ctx else False
	min_stiffen_anchor=ctx['min_stiffen_anchor'] if 'min_stiffen_anchor' in ctx else -1
	m_init_file=ctx['m_init_file'] if 'm_init_file' in ctx else None
	# newton_tol=1e-5
	newton_tol=1e-3
	cg_tol=1e-3
	cg_max_iter=1000
	print('use_constant_k',use_constant_k)

	stiffen_anchor=load_anchor_data(shared_data_dir,m_init_file=m_init_file)
	edges,l0,k=load_spring_data(shared_data_dir,use_constant_k)
	axial_i,axial_w,axial_k=load_axial_data(shared_data_dir)
	print('avg k',np.mean(k))

	stiffen_anchor=torch.from_numpy(stiffen_anchor).to(device=device,dtype=dtype)*stiffen_anchor_factor
	edges=torch.from_numpy(edges).to(device=device,dtype=torch.int32)
	l0=torch.from_numpy(l0).to(device=device,dtype=dtype)
	k=torch.from_numpy(k).to(device=device,dtype=dtype)
	axial_i=torch.from_numpy(axial_i).to(device=device,dtype=torch.int32)
	axial_w=torch.from_numpy(axial_w).to(device=device,dtype=dtype)
	axial_k=torch.from_numpy(axial_k).to(device=device,dtype=dtype)
	n_vts=len(stiffen_anchor)
	print('init_cudaqs_module:n_vts:',n_vts)

	module=CudaqsModule()
	cudaqs.init(n_vts,batch_size,use_multi_thread,verbose,min_stiffen_anchor)
	spring_data=cudaqs.init_spring(edges,l0,k)
	axial_data=cudaqs.init_axial(axial_i,axial_w,axial_k)
	system=cudaqs.init_system(n_vts,spring_data,axial_data)
	forward_opt=cudaqs.init_forward(system,newton_tol,cg_tol,cg_max_iter)
	backward_opt=cudaqs.init_backward(system,use_variable_m,cg_tol,cg_max_iter)
	# print('len(edges):',len(edges))
	opt_data=cudaqs.init_opt_data(batch_size,n_vts,len(edges))
	module.init(spring_data,axial_data,system,forward_opt,backward_opt,opt_data,use_variable_m)
	module.stiffen_anchor=stiffen_anchor.unsqueeze(0)

	print('init_cudaqs_module finish')

	return module

class CudaqsUtils:
	def __init__(self,shared_data_dir,batch_size=1,stiffen_anchor_factor=0.2):
		self.batch_size=batch_size
		self.shared_data_dir=shared_data_dir
		self.device=torch.device('cuda:0')
		self.dtype=torch.double
		res_ctx={'shared_data_dir':self.shared_data_dir}
		ctx={'batch_size':batch_size,'dtype':self.dtype,'device':self.device,'patch_id':-1,'max_num_constraints':-1,'verbose':True,'use_variable_m':False,'stiffen_anchor_factor':stiffen_anchor_factor,'use_constant_k':True}
		self.res_ctx=res_ctx
		self.ctx=ctx
		self.module=init_cudaqs_module(res_ctx,ctx)

	def forward(self,x):
		v=torch.from_numpy(x).to(device=self.device,dtype=self.dtype)
		v=v.unsqueeze(0).repeat(self.batch_size,1,1)
		anchor=v
		x=self.module(anchor)
		x=x[0].detach().cpu().numpy()
		return x

def filter_edges(vt_id_set,edges,aux_data):
	filtered_edges=[]
	filtered_aux_data=[]
	for edge,data in zip(edges,aux_data):
		if all([i in vt_id_set for i in edge]):
			filtered_edges.append(edge)
			filtered_aux_data.append(data)
	filtered_edges=np.array(filtered_edges)
	filtered_aux_data=np.array(filtered_aux_data)
	print('filtered_edges',filtered_edges.shape)
	return filtered_edges,filtered_aux_data

def filter_front_springs(src_dir,tgt_dir):
	front_vt_ids=np.loadtxt(join(src_dir,'front_vertices.txt')).astype(int)
	front_vt_id_set=set(front_vt_ids)
	def filter(edges_name,aux_name):
		edges=np.loadtxt(join(src_dir,edges_name)).astype(int)
		rest_lengths=np.loadtxt(join(src_dir,aux_name))
		edges,rest_lengths=filter_edges(front_vt_id_set,edges,rest_lengths)
		out_path=join(tgt_dir,edges_name)
		print('write to',out_path)
		np.savetxt(out_path,edges)
		out_path=join(tgt_dir,aux_name)
		print('write to',out_path)
		np.savetxt(join(tgt_dir,aux_name),rest_lengths)
	filter('linear_edges.txt','mat_or_med_linear.txt')
	filter('bend_edges.txt','mat_or_med_bend.txt')
	# filter('axial_spring_particles.txt','axial_particle_weights.txt')

if __name__=='__main__':
	filter_front_springs('../../pixel_network/shared_data_highres','../../pixel_network/shared_data_hr_front')


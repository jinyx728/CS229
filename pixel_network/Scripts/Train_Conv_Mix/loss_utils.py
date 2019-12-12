######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
from collections import defaultdict
import torch
import torch.nn as nn
class LossManager:
	def __init__(self,ctx):
		self.loss_type=ctx['loss_type']
		if self.loss_type=='mse':
			self.pix_loss_fn=nn.MSELoss(reduction='sum')
		elif self.loss_type=='l1':
			self.pix_loss_fn=nn.L1Loss(reduction='sum')
		self.clear()

	def add_item_loss(self,name,v):
		self.item_loss[name]+=v

	def add_total_loss(self,v):
		self.total_loss+=v

	def add_samples(self,num_samples):
		self.total_samples+=num_samples

	def clear(self):
		self.item_loss=defaultdict(float)
		self.total_loss=0
		self.total_samples=0

	def get_total_loss(self):
		return self.total_loss/self.total_samples

	def get_item_loss(self):
		item_loss={name:loss/self.total_samples for name,loss in self.item_loss.items()}
		return item_loss

	def get_pix_loss_both_sides(self,pd_offset_imgs,gt_offset_imgs,front_masks,back_masks,normalize=False):
		front_pd=pd_offset_imgs[:,:2,:,:]
		front_gt=gt_offset_imgs[:,:2,:,:]
		back_pd=pd_offset_imgs[:,2:,:,:]
		back_gt=gt_offset_imgs[:,2:,:,:]
		n_samples=len(gt_offset_imgs)

		if self.loss_type=='l2':
			# assume all masks are the same
			front_loss=torch.sum((((front_pd-front_gt)*front_masks)**2).view(n_samples,-1),dim=1)
			back_loss=torch.sum((((back_pd-back_gt)*back_masks)**2).view(n_samples,-1),dim=1)
			# return torch.mean(torch.sqrt(front_loss+back_loss))/torch.sqrt(torch.sum(front_masks[0]+back_masks[0]))
			return torch.mean(torch.sqrt(front_loss+back_loss))
			
		elif self.loss_type=='mse' or self.loss_type=='l1':
			front_loss=self.pix_loss_fn(front_pd*front_masks,front_gt*front_masks)
			back_loss=self.pix_loss_fn(back_pd*back_masks,back_gt*back_masks)
			if normalize:
				return (front_loss+back_loss)/(torch.sum(front_masks[0,0,:,:])+torch.sum(back_masks[0,0,:,:]))/n_samples
			else:
				return (front_loss+back_loss)/n_samples


	def get_pix_loss(self,pd_offset_imgs,gt_offset_imgs,masks,normalize=False):
		if self.loss_type=='l2':
			# n_samples=len(gt_offset_imgs)
			# loss=torch.sum((((pd_offset_imgs-gt_offset_imgs)*masks)**2).view(n_samples,-1),dim=1)
			# return torch.mean(torch.sqrt(loss))/torch.sqrt(torch.sum(masks[0]))
			return None
		elif self.loss_type=='mse' or self.loss_type=='l1':
			loss=self.pix_loss_fn(pd_offset_imgs*masks,gt_offset_imgs*masks)
			if normalize:
				return loss/torch.sum(masks)
			else:
				return loss

	def get_vt_loss(self,pd_vt,gt_vt):
		assert(pd_vt.size(2)==3)
		assert(gt_vt.size(2)==3)
		# return torch.mean(torch.norm(pd_vt-gt_vt,dim=2))
		n_samples,n_vts=gt_vt.size(0),gt_vt.size(1)
		return self.pix_loss_fn(pd_vt,gt_vt)/(n_vts*n_samples)

	def get_mse_loss(self,src,tgt,norm_vts=True):
		if norm_vts:
			mseloss=nn.MSELoss()
			return mseloss(src,tgt)
		else:
			mseloss=nn.MSELoss(reduction='sum')
			n_samples=len(src)
			return mseloss(src,tgt)/n_samples

	def get_avg_mse_loss(self,src_list,tgt,norm_vts=False):
		mseloss=nn.MSELoss(reduction='sum')
		avg_loss=0
		for src in src_list:
			avg_loss+=mseloss(src,tgt)
		avg_loss/=len(src_list)
		if norm_vts:
			avg_loss/=src_list[0].size(1)
		return avg_loss

	def print_item_loss(self,item_loss):
		loss_str=','.join(['{}:{:.6E}'.format(k,v) for k,v in item_loss.items()])
		print(loss_str)
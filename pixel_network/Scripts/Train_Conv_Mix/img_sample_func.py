######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import torch.nn as nn
from torch.autograd import Function
from offset_img_utils import OffsetManager
import numpy as np
import gzip

class ImgSampleBothSidesFunction(Function):

	@staticmethod
	def forward(ctx,offset_imgs,offset_manager):
		ctx.offset_manager=offset_manager
		return offset_manager.get_offsets_from_offset_imgs_both_sides(offset_imgs)

	@staticmethod
	def backward(ctx,grad_output):
		offset_manager=ctx.offset_manager
		mask_sum=offset_manager.mask_sum
		n_vts=grad_output.size(1)
		offset_imgs_grad=offset_manager.get_offset_imgs_from_offsets_both_sides(grad_output*(n_vts/mask_sum))
		return offset_imgs_grad,None

img_sample_both_sides=ImgSampleBothSidesFunction.apply

class ImgSampleBothSidesModule(nn.Module):
	def __init__(self,offset_manager):
		super(ImgSampleBothSidesModule,self).__init__()
		self.offset_manager=offset_manager

	def forward(self,offset_imgs):
		return img_sample_both_sides(offset_imgs,self.offset_manager)

class ImgSample2ndOrderBothSidesFunction(Function):

	@staticmethod
	def forward(ctx,offset_imgs,offset_manager):
		ctx.offset_manager=offset_manager
		with torch.no_grad():
			offsets=offset_manager.get_offsets_from_offset_imgs_both_sides(offset_imgs)
		ctx.save_for_backward(offset_imgs,offsets)
		offsets.requires_grad_(True)
		return offsets

	@staticmethod
	def backward(ctx,grad_output):
		offset_manager=ctx.offset_manager
		input_offset_imgs,offsets=ctx.saved_tensors
		mask=offset_manager.mask
		mask_sum=offset_manager.mask_sum
		n_vts=grad_output.size(1)
		with torch.no_grad():
			approx_gt_imgs=offset_manager.get_offset_imgs_from_offsets_both_sides(offsets-grad_output*(n_vts/mask_sum))
			grad_imgs=input_offset_imgs-approx_gt_imgs
			# grad_imgs=offset_manager.get_offset_imgs_from_offsets_both_sides(grad_output)
			grad_imgs[:,:3,:,:]*=mask[:,:,0].view(1,1,mask.size(0),mask.size(1))
			grad_imgs[:,3:,:,:]*=mask[:,:,1].view(1,1,mask.size(0),mask.size(1))
			# print('grad_output',torch.min(grad_output).item(),torch.max(grad_output).item(),torch.mean(torch.abs(grad_output)).item(),'grad_imgs',torch.min(grad_imgs).item(),torch.max(grad_imgs).item(),torch.mean(torch.abs(grad_imgs)).item(),'input',torch.min(input_offset_imgs).item(),torch.max(input_offset_imgs).item())
		return grad_imgs,None

img_sample_2nd_order_both_sides=ImgSample2ndOrderBothSidesFunction.apply

class ImgSample2ndOrderBothSidesModule(nn.Module):
	def __init__(self,offset_manager):
		super(ImgSample2ndOrderBothSidesModule,self).__init__()
		self.offset_manager=offset_manager

	def forward(self,offset_imgs):
		return img_sample_2nd_order_both_sides(offset_imgs,self.offset_manager)


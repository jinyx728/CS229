######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Train_Conv_Mix')
import os
from os.path import join,isdir
import torch
import numpy as np
import gzip
from offset_img_utils import OffsetManager
from img_sample_func import ImgSampleBothSidesModule,ImgSample2ndOrderBothSidesModule
from render_offsets import save_png_img,get_png_mask_both_sides

def normalize_img(img,v=None):
	if v is None:
		v=np.max(np.abs(img))
	return img/(2*v)+0.5

class ImgSampleTest:
	def __init__(self):
		self.shared_data_dir='../../shared_data'
		self.data_root_dir='/data/zhenglin/poses_v3'
		self.offset_img_size=128
		self.device=torch.device('cuda:0')
		self.dtype=torch.double
		self.res='lowres'
		self.offset_manager=OffsetManager(shared_data_dir=self.shared_data_dir,ctx={'data_root_dir':self.data_root_dir,'offset_img_size':self.offset_img_size,'device':self.device,'dtype':self.dtype})
		# self.img_sample_both_sides=ImgSampleBothSidesModule(self.offset_manager)
		self.img_sample_both_sides=ImgSample2ndOrderBothSidesModule(self.offset_manager)
		# self.pd_offset_img_dir='opt_test'
		self.pd_offset_img_dir='../../rundir/lowres/xyz/eval_test'
		self.gt_offset_dir=join(self.data_root_dir,'{}_offset_npys'.format(self.res))
		self.gt_offset_img_dir=join(self.data_root_dir,'{}_offset_imgs_{}'.format(self.res,self.offset_img_size))
		self.out_img_dir='opt_test/img_sample'
		if not isdir(self.out_img_dir):
			os.makedirs(self.out_img_dir)

		mask=np.load(join(self.shared_data_dir,'offset_img_mask_{}.npy'.format(self.offset_img_size)))
		# print('mask',np.unique(mask))
		self.png_mask=get_png_mask_both_sides(mask)
		self.mask_sum=np.sum(mask)

	def img_test(self,sample_id):
		offset=np.load(join(self.gt_offset_dir,'offset_{:08d}.npy'.format(sample_id)))
		offsets=torch.from_numpy(offset).to(device=self.device,dtype=self.dtype).unsqueeze(0)
		offset_imgs=self.offset_manager.get_offset_imgs_from_offsets_both_sides(offsets)
		offset_img=offset_imgs[0].permute(1,2,0).cpu().numpy()
		with gzip.open(join('opt_test/img_sample','offset_img_{:08d}.npy.gz'.format(sample_id)),'wb') as f:
			np.save(file=f,arr=offset_img)
		save_png_img(join(self.out_img_dir,'ras_offset_img.png'),normalize_img(offset_img))

		with gzip.open(join(self.gt_offset_img_dir,'offset_img_{:08d}.npy.gz'.format(sample_id)),'rb') as f:
			gt_offset_img=np.load(file=f)
		save_png_img(join(self.out_img_dir,'gt_offset_img.png'),normalize_img(gt_offset_img))

		diff=offset_img-gt_offset_img
		print('diff:norm:',np.linalg.norm(diff),'max:',np.max(np.abs(diff)))
		# difference are on the boundary
		save_png_img(join(self.out_img_dir,'diff_offset_img.png'),normalize_img(offset_img-gt_offset_img))

		sample_offsets=self.offset_manager.get_offsets_from_offset_imgs_both_sides(offset_imgs)
		diff_offsets=sample_offsets-offsets
		print('diff_offset,norm:',torch.norm(diff_offsets).item(),'max:',torch.max(torch.abs(diff_offsets)).item(),'max_ratio:',torch.max(torch.norm(diff_offsets,dim=1)/torch.norm(offsets,dim=1)).item())
		# print('diff_offset:',torch.norm(diff_offsets[0],dim=1).detach().cpu().numpy().tolist())
		
		sample_offset_imgs=self.offset_manager.get_offset_imgs_from_offsets_both_sides(sample_offsets)
		diff2=sample_offset_imgs-offset_imgs
		print('diff2:norm:',torch.norm(diff2).item(),'max:',torch.max(torch.abs(diff2)).item())
		diff2_img=diff2[0].detach().permute(1,2,0).cpu().numpy()
		save_png_img(join(self.out_img_dir,'diff2_offset_img.png'),normalize_img(diff2_img))


	def test(self,sample_id):
		with gzip.open(join(self.pd_offset_img_dir,'pd_img_{:08d}.npy.gz'.format(sample_id))) as f:
			pd_offset_img=np.load(file=f)
		pd_offset_imgs=torch.from_numpy(pd_offset_img).permute(2,0,1).to(self.device,dtype=self.dtype).unsqueeze(0)
		print('pd_offset_imgs',pd_offset_imgs.size())
		gt_offset=np.load(join(self.gt_offset_dir,'offset_{:08d}.npy'.format(sample_id)))
		gt_offsets=torch.from_numpy(gt_offset).to(self.device,dtype=self.dtype).unsqueeze(0)
		loss_fn=torch.nn.MSELoss()

		pd_offset_imgs.requires_grad_(True)
		pd_offsets=self.offset_manager.get_offsets_from_offset_imgs_both_sides(pd_offset_imgs)
		loss=loss_fn(pd_offsets,gt_offsets)
		loss.backward()
		grad=pd_offset_imgs.grad[0].permute(1,2,0).cpu().numpy()
		print('dir grad,min',np.min(grad),'max',np.max(grad))
		save_png_img(join(self.out_img_dir,'dir_grad.png'),normalize_img(grad),self.png_mask)
		v=np.max(np.abs(grad))

		pd_offset_imgs.grad=None
		pd_offset_imgs.requires_grad_(True)
		pd_offsets=self.img_sample_both_sides(pd_offset_imgs)
		loss=loss_fn(pd_offsets,gt_offsets)
		loss.backward()
		grad=pd_offset_imgs.grad[0].permute(1,2,0).cpu().numpy()
		print('sample grad,min',np.min(grad),'max',np.max(grad))
		g=normalize_img(grad,v)
		print('g,min',np.min(g),'max',np.max(g))
		# save_png_img(join(self.out_img_dir,'sample_grad.png'),normalize_img(grad),self.png_mask)
		save_png_img(join(self.out_img_dir,'sample_grad.png'),normalize_img(grad),self.png_mask)

	def numeric_test(self,sample_id):
		pd_offset_imgs=torch.zeros((1,6,self.offset_img_size,self.offset_img_size),device=self.device,dtype=self.dtype)
		pd_offset_imgs.requires_grad_(True)
		pd_offsets=self.img_sample_both_sides(pd_offset_imgs)
		pd_offsets.requires_grad_(True)
		print('pd_offsets',np.unique(pd_offsets[0].detach().cpu().numpy()))
		gt_offsets=torch.ones_like(pd_offsets,device=self.device,dtype=self.dtype)
		loss_fn=torch.nn.MSELoss(reduction='sum')
		loss=loss_fn(pd_offsets,gt_offsets)
		# loss=torch.sum((gt_offsets-pd_offsets)**2)
		print('loss',loss.item())
		loss.backward()
		# offset_grad=pd_offsets.grad[0].cpu().numpy()
		# print('offset_grad',np.unique(offset_grad))
		img_grad=pd_offset_imgs.grad[0].permute(1,2,0).cpu().numpy()
		print('img_grad',np.unique(img_grad))
		offset_manager=self.img_sample_both_sides.offset_manager
		print('w min',torch.min(offset_manager.vt_ws_img).item(),'max',torch.max(offset_manager.vt_ws_img).item())
		front_wsum_img,back_wsum_img=torch.sum(offset_manager.vt_ws_img[:,:,:3],dim=2),torch.sum(offset_manager.vt_ws_img[:,:,3:],dim=2)
		print('front_ws',np.unique(front_wsum_img.cpu().numpy()))
		print('back_ws',np.unique(back_wsum_img.cpu().numpy()))
		print('mask',np.unique(offset_manager.mask.cpu().numpy()))

		save_png_img(join(self.out_img_dir,'numerics_grad.png'),normalize_img(img_grad),self.png_mask)

	def vt_test(self,sample_id):
		with gzip.open(join(self.gt_offset_img_dir,'offset_img_{:08d}.npy.gz'.format(sample_id))) as f:
			gt_offset_img=np.load(file=f)
		gt_offset_imgs=torch.from_numpy(gt_offset_img).permute(2,0,1).to(self.device,dtype=self.dtype).unsqueeze(0)
		gt_offset=np.load(join(self.gt_offset_dir,'offset_{:08d}.npy'.format(sample_id)))
		gt_offsets=torch.from_numpy(gt_offset).to(self.device,dtype=self.dtype).unsqueeze(0)

		ras_gt_offset_imgs=self.offset_manager.get_offset_imgs_from_offsets_both_sides(gt_offsets)
		sample_ras_offsets=self.offset_manager.get_offsets_from_offset_imgs_both_sides(ras_gt_offset_imgs)
		sample_offsets=self.offset_manager.get_offsets_from_offset_imgs_both_sides(gt_offset_imgs)

		print('sample error',torch.norm(sample_offsets-gt_offsets),'ras sample error',torch.norm(sample_ras_offsets-gt_offsets))

	def grad_test(self,sample_id):
		with gzip.open(join(self.pd_offset_img_dir,'{:08d}/pd_img_{:08d}.npy.gz'.format(sample_id,sample_id)),'rb') as f:
			pd_offset_img=np.load(file=f)
		pd_offset_imgs=torch.from_numpy(pd_offset_img).permute(2,0,1).to(device=self.device,dtype=self.dtype).unsqueeze(0)
		with gzip.open(join('opt_test/img_sample','offset_img_{:08d}.npy.gz'.format(sample_id))) as f:
			gt_offset_img=np.load(file=f)
		gt_offset_imgs=torch.from_numpy(gt_offset_img).permute(2,0,1).to(self.device,dtype=self.dtype).unsqueeze(0)
		mseloss=torch.nn.MSELoss(reduction='sum')
		pd_offset_imgs.requires_grad_(True)
		print('pd_offset_imgs',pd_offset_imgs.size())
		gt_loss=mseloss(pd_offset_imgs,gt_offset_imgs)/self.mask_sum
		gt_loss.backward()
		gt_grad=pd_offset_imgs.grad.clone()

		pd_offset_imgs.grad.zero_()
		pd_offset_imgs.requires_grad_(True)
		with open(join(self.gt_offset_dir,'offset_{:08d}.npy'.format(sample_id)),'rb') as f:
			gt_offset=np.load(file=f)
		n_vts=len(gt_offset)
		gt_offsets=torch.from_numpy(gt_offset).to(device=self.device,dtype=self.dtype).unsqueeze(0)
		pd_offsets=self.img_sample_both_sides(pd_offset_imgs)
		test_loss=mseloss(pd_offsets,gt_offsets)/n_vts
		print('gt_loss',gt_loss.item(),'test_loss',test_loss.item(),'n_vts',n_vts,'mask_sum',self.mask_sum)
		test_loss.backward()
		test_grad=pd_offset_imgs.grad

		diff_grad=test_grad-gt_grad
		diff_norm=torch.norm(diff_grad).item()
		diff_max=torch.max(diff_grad).item()
		print('norm1:',torch.norm(gt_grad).item(),'norm2:',torch.norm(test_grad).item())
		print('diff:norm',diff_norm,'max:',diff_max)
		diff_img=diff_grad[0].detach().permute(1,2,0).cpu().numpy()
		save_png_img(join(self.out_img_dir,'diff_grad2.png'),normalize_img(diff_img),self.png_mask)




if __name__=='__main__':
	test=ImgSampleTest()
	# test.test(106)
	# test.img_test(106)
	# test.numeric_test(106)
	# test.vt_test(106)
	test.grad_test(106)
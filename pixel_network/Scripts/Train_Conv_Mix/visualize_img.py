######################################################################
# Copyright 2019. Yongxu Jin.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Data_Processing')
import os 
from os.path import join
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from image_dataset_utils import TshirtImageDataset
from mix_data_loader import MixDataLoader
from torch.utils.data import Dataset, DataLoader
from conv_decoder import ConvDecoder
from train_utils import count_parameters,print_layer_parameters,train_model,eval_model
from ctx_utils import ctx
from vlz_utils import from_offset_img_to_rgb_img
from offset_img_utils import OffsetManager
import gzip
# from torchsummary import summary

input_rotate=torch.from_numpy(np.load('/data/yxjin/poses_v3/rotation_matrices/rotation_mat_00015088.npy')).float()

net=ConvDecoder(ctx['input_size'],
                init_linear_layers=ctx['init_linear_layers'],
                output_size=ctx['offset_img_size'] if not ctx['use_patches'] else ctx['crop_size'],
                use_coord_conv=ctx['use_coord_conv'],
                use_up_conv=ctx['use_up_conv'],
                use_skip_link=ctx['use_skip_link'],
                use_multi_layer_loss=ctx['use_multi_layer_loss'],
                init_channels=ctx['init_channels'],
                output_channels=ctx['output_channels'],
                init_size=ctx['init_size'],
                n_res_blocks=ctx['n_res_blocks'],
                use_dropout=ctx['use_dropout'],
                relu_type=ctx['relu'])

cp=torch.load(ctx['cp'],map_location='cuda:0')
net.load_state_dict(cp['state_dict'])

# summary
# model = net.to('cuda:0')
# summary(model, input_rotate.shape)

# visualization
# offset_img=net(input_rotate).detach().numpy()[0,:,:,:]
# mask_path='/data/yxjin/PhysBAM/Private_Projects/cloth_texture/pixel_network/Learning/shared_data_lowres/offset_img_mask_512.npy'
# mask=torch.from_numpy(np.load(mask_path)).squeeze().to(device='cuda:0',dtype=torch.double)
# front_mask=(mask[:,:,0].view(1,1,mask.size(0),mask.size(1)))[0].squeeze().cpu().numpy()
# back_mask=(mask[:,:,1].view(1,1,mask.size(0),mask.size(1)))[0].squeeze().cpu().numpy()
# gt_path='/data/yxjin/poses_v3/lowres_texture_imgs_512/offset_img_00015001.npy.gz'
# gt_img=np.load(gzip.open(gt_path))

# arr_stats={'minval':np.full(6,np.min(gt_img)),'maxval':np.full(6,np.max(gt_img))}

# front_offset_img=np.transpose(offset_img[:2,:,:],(1,2,0))
# gt_front_offset_img=gt_img[:,:,:2]
# front_arr_stats={name:stats[:3] for name,stats in arr_stats.items()}
# gt_front_rgb_img=from_offset_img_to_rgb_img(gt_front_offset_img,front_mask,arr_stats=front_arr_stats)[:, 0:1024]
# front_rgb_img=from_offset_img_to_rgb_img(front_offset_img,front_mask,arr_stats=front_arr_stats)[:, 0:1024]

# back_offset_img=np.transpose(offset_img[2:4,:,:],(1,2,0))
# gt_back_offset_img=gt_img[:,:,2:4]
# back_arr_stats={name:stats[3:] for name,stats in arr_stats.items()}
# gt_back_rgb_img=from_offset_img_to_rgb_img(gt_back_offset_img,back_mask,arr_stats=back_arr_stats)[:, 0:1024]
# back_rgb_img=from_offset_img_to_rgb_img(back_offset_img,back_mask,arr_stats=back_arr_stats)[:, 0:1024]

# output_img=np.concatenate([front_rgb_img, back_rgb_img], axis=1)
# gt_output_img=np.concatenate([gt_front_rgb_img, gt_back_rgb_img], axis=1)
# output_img=np.concatenate([gt_output_img,output_img],axis=0)
# im = Image.fromarray(output_img)
# im.save("output_img.jpeg")

# convert back to vertex displacement
offset_manager=OffsetManager(shared_data_dir='/data/yxjin/PhysBAM/Private_Projects/cloth_texture/pixel_network/Learning/shared_data_highres',ctx=ctx)
predict=net(input_rotate)
pd_vt_offsets=offset_manager.get_offsets_from_offset_imgs_both_sides(predict)
np.savetxt('displace_test.txt', pd_vt_offsets[0,:,:].detach().numpy())
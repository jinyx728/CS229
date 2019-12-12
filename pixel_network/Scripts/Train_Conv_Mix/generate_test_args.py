######################################################################
# Copyright 2019. Yongxu Jin.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Data_Processing')
import os 
from os.path import join,isdir
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

print('sample_list_file',ctx['sample_list_file'])
print('data_root_dir',ctx['data_root_dir'])
print('eval_out_dir',ctx['eval_out_dir'])
# samples = np.loadtxt('/data/yxjin/poses_v3/sample_lists/lowres_test_samples.txt')
# samples = np.loadtxt('/data/yxjin/poses_v3/sample_lists/highres_train_samples.txt')
samples=np.loadtxt(ctx['sample_list_file']).astype(int)
if not isdir(ctx['eval_out_dir']):
    os.makedirs(ctx['eval_out_dir'])

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

offset_manager=OffsetManager(shared_data_dir=ctx['res_ctx']['shared_data_dir'],ctx=ctx)

for sample in samples:
    index = '{:08d}'.format(int(sample))
    input_rotate=torch.from_numpy(np.load(join(ctx['data_root_dir'],'rotation_matrices/rotation_mat_{}.npy'.format(index)))).float()
    predict=net(input_rotate)
    pd_vt_offsets=offset_manager.get_offsets_from_offset_imgs_both_sides(predict)
    np.savetxt(join(ctx['eval_out_dir'],'displace_{}.txt'.format(index)), pd_vt_offsets[0,:,:].detach().numpy())
    print('generated data {}'.format(index))
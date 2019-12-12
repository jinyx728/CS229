######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
# #Import
import sys
sys.path.insert(0,'../Data_Processing')
import os 
from os.path import join
import numpy as np
import torch
import torch.nn as nn
from image_dataset_utils import TshirtImageDataset
from mix_data_loader import MixDataLoader
from torch.utils.data import Dataset, DataLoader
from conv_decoder import ConvDecoder
from train_utils import count_parameters,print_layer_parameters,train_model,eval_model
from ctx_utils import ctx
# datasets
datasets={}
dataloaders={}

if not ctx['use_mix']:
    if ctx['eval']=='none':
        dataset_names=['train','val']
    else:
        dataset_names=[ctx['eval']]

    res_ctx=ctx['res_ctx']
    for name in dataset_names:
        dataset=TshirtImageDataset(sample_list_file=res_ctx['sample_list_file_dict'][name],res_ctx=res_ctx,ctx=ctx)
        datasets[name]=dataset

else:
    if ctx['eval']=='none':
        for res_name,res_ctx in ctx['mixres_ctxs'].items():
            datasets['train_{}'.format(res_name)]=TshirtImageDataset(sample_list_file=res_ctx['sample_list_file_dict']['train'],res_ctx=res_ctx,ctx=ctx)
        res_ctx=ctx['mixres_ctxs']['midres']
        datasets['val']=TshirtImageDataset(sample_list_file=res_ctx['sample_list_file_dict']['val'],res_ctx=res_ctx,ctx=ctx)
    else:
        res_ctx=ctx['mixres_ctxs']['midres']
        datasets[ctx['eval']]=TshirtImageDataset(sample_list_file=res_ctx['sample_list_file_dict'][ctx['eval']],res_ctx=res_ctx,ctx=ctx)

for name,dataset in datasets.items():
    dataloaders[name]=DataLoader(dataset,batch_size=ctx['batch_size'],shuffle=not ctx['no_shuffle'],num_workers=ctx['num_workers'])
    print(name,'num samples',len(dataset))

if ctx['use_mix'] and ctx['eval']=='none':
    dataloaders['train']=MixDataLoader({'midres':dataloaders['train_midres'],'lowres':dataloaders['train_lowres']})

# network
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
print('learning_rate',ctx['lr'],'weight_decay',ctx['weight_decay'])
print('sample_list_file',ctx['sample_list_file'],'max_num_samples',ctx['max_num_samples'])
opt=torch.optim.Adam(net.parameters(), lr=ctx['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=ctx['weight_decay'], amsgrad=False)

print(net)
print('total parameters',count_parameters(net))
print_layer_parameters(net)
net=net.double().to(ctx['device'])

if ctx['use_variable_m']:
    m_net=ConvDecoder(ctx['input_size'],
                init_linear_layers=ctx['init_linear_layers'],
                output_size=ctx['offset_img_size'] if not ctx['use_patches'] else ctx['crop_size'],
                use_coord_conv=ctx['use_coord_conv'],
                use_up_conv=ctx['use_up_conv'],
                use_skip_link=ctx['use_skip_link'],
                use_multi_layer_loss=ctx['use_multi_layer_loss'],
                init_channels=ctx['m_init_channels'],
                output_channels=ctx['m_output_channels'],
                init_size=ctx['init_size'],
                n_res_blocks=ctx['n_res_blocks'],
                use_dropout=ctx['use_dropout'],
                relu_type=ctx['relu'])
    m_opt=torch.optim.Adam(m_net.parameters(), lr=ctx['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=ctx['weight_decay'], amsgrad=False)
    print(m_net)
    print('total parameters',count_parameters(m_net))
    print_layer_parameters(m_net)
    ctx['m_net']=m_net.double().to(ctx['device'])
    ctx['m_opt']=m_opt

# Load checkpoint
start_epoch=0
if ctx['cp'] != '':
    print('Load checkpoint',ctx['cp'])
    cp=torch.load(ctx['cp'],map_location={"cuda:{}".format(i):"cuda:0" for i in range(8)})
    print('trained for', cp['epoch'], 'epochs', 'val cost',cp['val_cost'])

    net.load_state_dict(cp['state_dict'])
    if ctx['load_opt']:
        opt.load_state_dict(cp['optimizer'])

    start_epoch=cp['epoch']+1

    if ctx['use_variable_m']:
        if 'm_net' in cp:
            ctx['m_net'].load_state_dict(cp['m_net'])
            if ctx['load_opt']:
                opt.load_state_dict(cp['optimizer'])
                if 'm_opt' in cp:
                    ctx['m_opt'].load_state_dict(cp['m_opt'])
        else:
            print('m_net not in the checkpoint!')

start_iter=0
ctx['start_iter']=0

# loss
if ctx['loss_type']=='l2' or ctx['loss_type']=='mse':
    loss_fn=nn.MSELoss(reduction='sum')
elif ctx['loss_type']=='l1':
    loss_fn=nn.L1Loss(reduction='sum')
print('loss_type',ctx['loss_type'])

# tensorboard
from logger import Logger
log_dir=join(ctx['rundir'],'logs')
print('log dir',log_dir)
ctx['tb_logger']=Logger(log_dir)

if ctx['write_file_log']:
    file_log_dir=join(ctx['rundir'],'file_logs')
    print('file log dir',file_log_dir)
    from file_logger import FileLogger
    ctx['file_logger']=FileLogger(file_log_dir)

# start train/eval
if ctx['eval']=='none': # train
    train_model(dataloaders,net,opt,loss_fn,start_epoch,ctx)
else:
    eval_model(dataloaders[ctx['eval']],net,loss_fn,ctx)

######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Data_Processing')
import os 
from os.path import join
import numpy as np
import torch
import torch.nn as nn
from image_dataset_utils import TshirtImageDataset
from mix_data_loader import MixDataLoader
from hero_data_loader import HeroDataLoader
from torch.utils.data import Dataset, DataLoader
import progressive_gan_utils as progressive_gan
import style_gan_utils as style_gan
from train_utils import count_parameters,print_layer_parameters
from train_progressive_gan_utils import train_model,load_model,eval_model
from ctx_utils import ctx

# datasets
datasets={}
dataloaders={}

if ctx['use_mix']:
    if ctx['eval']=='none':
        for res_name,res_ctx in ctx['mixres_ctxs'].items():
            datasets['train_{}'.format(res_name)]=TshirtImageDataset(sample_list_file=res_ctx['sample_list_file_dict']['train'],res_ctx=res_ctx,ctx=ctx)
        res_ctx=ctx['mixres_ctxs']['midres']
        datasets['val']=TshirtImageDataset(sample_list_file=res_ctx['sample_list_file_dict']['val'],res_ctx=res_ctx,ctx=ctx)
    else:
        res_ctx=ctx['mixres_ctxs']['midres']
        datasets[ctx['eval']]=TshirtImageDataset(sample_list_file=res_ctx['sample_list_file_dict'][ctx['eval']],res_ctx=res_ctx,ctx=ctx)
elif ctx['use_hero']:
    res_ctx=ctx['res_ctx']
    sim_res_ctx=ctx['sim_res_ctx']
    if ctx['eval']=='none':
        for name in ['train','val']:
            datasets[name]=TshirtImageDataset(sample_list_file=res_ctx['sample_list_file_dict'][name],res_ctx=res_ctx,ctx=ctx)
            sim_dataset=TshirtImageDataset(sample_list_file=sim_res_ctx['sample_list_file_dict'][name],res_ctx=sim_res_ctx,ctx=ctx)
            datasets[name].sim_dataset=sim_dataset
    elif ctx['eval'].startswith('hero'):
        name=ctx['eval'][5:]
        datasets[ctx['eval']]=TshirtImageDataset(sample_list_file=res_ctx['sample_list_file_dict'][name],res_ctx=res_ctx,ctx=ctx)
    else:
        name=ctx['eval']
        datasets[ctx['eval']]=TshirtImageDataset(sample_list_file=res_ctx['sample_list_file_dict'][name],res_ctx=res_ctx,ctx=ctx)
else:
    if ctx['eval']=='none':
        dataset_names=['train','val']
    else:
        dataset_names=[ctx['eval']]

    res_ctx=ctx['res_ctx']
    for name in dataset_names:
        dataset=TshirtImageDataset(sample_list_file=res_ctx['sample_list_file_dict'][name],res_ctx=res_ctx,ctx=ctx)
        datasets[name]=dataset

for name,dataset in datasets.items():
    dataloaders[name]=DataLoader(dataset,batch_size=ctx['batch_size'],shuffle=not ctx['no_shuffle'],num_workers=ctx['num_workers'],pin_memory=True)
    print(name,'num samples',len(dataset))

if ctx['use_mix'] and ctx['eval']=='none':
    dataloaders['train']=MixDataLoader({'midres':dataloaders['train_midres'],'lowres':dataloaders['train_lowres']})


# networks
if ctx['use_ctgr_D']:
    D_output_size=2
elif ctx['use_ctgr_D_v2']:
    D_output_size=3
else:
    D_output_size=1
nets={}
if ctx['use_style_gan']:
    nets['G']=style_gan.G(start_level=ctx['start_level'],end_level=ctx['end_level'],input_size=ctx['input_size'],output_channels=6,channels=ctx['channelsG'],use_coord_conv=ctx['use_coord_conv'],ctx=ctx)
    nets['D']=style_gan.D(start_level=ctx['start_level'],end_level=ctx['end_level'],input_channels=6,channels=ctx['channelsD'],cond_size=ctx['input_size'],use_coord_conv=ctx['use_coord_conv'],ctx=ctx)
else:
    nets['G']=progressive_gan.G(start_level=ctx['start_level'],end_level=ctx['end_level'],input_size=ctx['input_size'],output_channels=6,channels=ctx['channelsG'])
    nets['D']=progressive_gan.D(start_level=ctx['start_level'],end_level=ctx['end_level'],input_channels=12 if ctx['cat_skin_imgs'] else 6,channels=ctx['channelsD'],output_size=D_output_size)
nets['G']=nets['G'].to(dtype=ctx['dtype']).to(device=ctx['device'])
nets['D']=nets['D'].to(dtype=ctx['dtype']).to(device=ctx['device'])

def init_opts(nets,lr=ctx['lr'],wd=ctx['weight_decay']):
    opts={}
    print('lr',lr)
    opts['G']=torch.optim.Adam(nets['G'].parameters(),lr=lr,betas=(0,0.9),eps=1e-8,weight_decay=wd)
    opts['D']=torch.optim.Adam(nets['D'].parameters(),lr=lr,betas=(0,0.9),eps=1e-8,weight_decay=wd)
    return opts
opts=init_opts(nets)
ctx['init_opts']=init_opts
print('G',nets['G'])
print_layer_parameters(nets['G'])
print('total parameters',count_parameters(nets['G']))
print('D',nets['D'])
print_layer_parameters(nets['D'])
print('total parameters',count_parameters(nets['D']))


# Load checkpoint
start_epoch=ctx['start_epoch']
if ctx['cp']!='':
    print('load checkpoint',ctx['cp'])
    cp=torch.load(ctx['cp'],map_location={"cuda:{}".format(i):"cuda:0" for i in range(8)})
    print('trained for',cp['epoch'],'epochs')

    load_model(nets,opts,cp)
    start_epoch=cp['epoch']
start_iter=0
ctx['start_iter']=0


# tensorboard
from logger import Logger
log_dir=join(ctx['rundir'],'logs')
print('log dir',log_dir)
ctx['tb_logger']=Logger(log_dir)

# start train/eval
if ctx['eval']=='none': # train
    train_model(dataloaders,nets,opts,start_epoch,ctx)
else:
    eval_model(dataloaders[ctx['eval']],nets,cp['epoch']-1,ctx)
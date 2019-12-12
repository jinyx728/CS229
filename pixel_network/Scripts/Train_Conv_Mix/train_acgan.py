######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################import sys
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
import acgan_utils as acgan
import acgan_regress_utils as acgan_regress
from train_utils import count_parameters,print_layer_parameters
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
    elif ctx['eval'].startswith('hero')!=-1:
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
nets={}
opts={}
if ctx['use_regress']:
    nets['R']=acgan.R()
    opts['R']=torch.optim.Adam(nets['R'].parameters(),lr=ctx['lr'],betas=(0.9,0.999))
    nets['R'].to(ctx['device'])

print('R',nets['R'])
print_layer_parameters(nets['R'])
print('total parameters',count_parameters(nets['R']))



# Load checkpoint
start_epoch=0
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
    if ctx['use_regress']:
        acgan_regress.train_model(dataloaders,nets['R'],opts['R'],start_epoch,ctx)
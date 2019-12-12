######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
# sys.path.insert(0,'../Data_Processing')
import torch
import torch.nn as nn
import numpy as np
import copy
import time
from datetime import datetime
import os
from os.path import join,exists
import shutil
from PIL import Image
from offset_img_utils import OffsetManager
from offset_io_utils import OffsetIOManager
from loss_utils import LossManager
from train_utils import set_requires_grad,load_model,save_model

def process_batch(batch,net,opt,managers,epoch,mode='train',res_ctx=None,ctx=None):
    device=ctx['device']
    dtype=ctx['dtype']
    gt_rotations=batch['rotations'].to(device=device,dtype=dtype,non_blocking=True)
    offset_imgs=batch['offset_img'].to(device=device,dtype=dtype,non_blocking=True)
    n_samples=len(gt_rotations)

    pd_rotations=net(offset_imgs)

    loss_manager=managers['loss_manager']

    loss=loss_manager.get_l2_loss(pd_rotations,gt_rotations)
    loss_manager.add_item_loss('l2_loss',loss.item()*n_samples)

    loss_manager.add_total_loss(loss.item()*n_samples,n_samples)

    # step 
    if mode=='train':
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss

def process_epoch(dataloader,net,opt,epoch,ctx,mode='train'):
    epoch_start_time=time.time()
    if mode=='train':
        net.train()
        torch.set_grad_enabled(True)
    else: # validate or evaluation
        net.eval()
        torch.set_grad_enabled(False)
    
    # managers and res_ctx
    def get_managers(res_ctx):
        managers={
            'loss_manager':LossManager(ctx=ctx)
        }
        return managers

    managers=get_managers(ctx['res_ctx'])
    res_ctx=ctx['res_ctx']

    # process batches
    for idx,batch in enumerate(dataloader):
        if ctx['use_mix'] and mode=='train':
            managers=res_managers[batch['label']]
            res_ctx=mixres_ctxs[batch['label']]

        process_batch(batch,net,opt,managers,epoch,mode=mode,res_ctx=res_ctx,ctx=ctx)

    # log loss
    if mode=='train' or mode=='validate':
        def log_loss(loss_manager,tb_logger,surfix=''):
            item_loss=loss_manager.get_item_loss()
            for name,loss in item_loss.items():
                tb_logger.scalar_summary('{}/{}{}'.format(mode,name,surfix),loss,epoch)

        log_loss(managers['loss_manager'],ctx['tb_logger'])

    # print necessary information
    epoch_end_time=time.time()
    total_loss=managers['loss_manager'].get_total_loss()
    print(mode,'epoch:',epoch,', time:',time.strftime('%H:%M:%S',time.gmtime(epoch_end_time-epoch_start_time)),'total_loss',total_loss)

    return total_loss

def train_model(dataloaders,net,opt,start_epoch,ctx):
    best_val_loss=float('Inf')

    best_train_loss=float('Inf')

    rundir=ctx['rundir']
    with open(join(rundir,'train_args.txt'),'w') as fout:
        for key,value in ctx.items():
            if isinstance(value,str) or isinstance(value,float) or isinstance(value,int) or isinstance(value,bool) or value is None:
                fout.write('{}:{}\n'.format(key,value))

    save_dir=join(rundir,'saved_models')
    if not exists(save_dir):
        os.makedirs(save_dir)

    save_every_epoch=ctx.get('save_every_epoch',10)
    num_epochs=ctx['num_epochs']
    print('start training from epoch',start_epoch,'auto save every',save_every_epoch,'epochs')

    for epoch in range(start_epoch,num_epochs):
        train_loss=process_epoch(dataloaders['train'],net,opt,epoch,ctx,mode='train')
        validate_loss=process_epoch(dataloaders['val'],net,opt,epoch,ctx,mode='validate')

        if validate_loss<best_val_loss:
            print('validate loss is lower than best before, saving model...')
            best_val_loss=validate_loss
            save_path=join(save_dir,'val_model_best.pth.tar')
            save_model(net,opt,ctx['lr'],validate_loss,epoch,save_path)
            print('save model:',save_path)

        if train_loss<best_train_loss:
            print('train loss is lower than best before, saving model...')
            best_train_loss=train_loss
            save_path=join(save_dir,'train_model_best.pth.tar')
            save_model(net,opt,ctx['lr'],train_loss,epoch,save_path)

        if (epoch+1)%save_every_epoch==0:
            save_path=join(save_dir,'checkpoint-{}.pth.tar'.format(epoch+1))
            save_model(net,opt,ctx['lr'],validate_loss,epoch,save_path)

def eval_model(dataloader,net,ctx):
    process_epoch(dataloader,net,opt=None,epoch=0,ctx=ctx,mode='eval')




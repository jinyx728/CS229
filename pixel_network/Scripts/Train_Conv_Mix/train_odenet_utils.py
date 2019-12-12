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
from os.path import join,exists,isdir
import shutil
from PIL import Image
from offset_img_utils import OffsetManager
from offset_io_utils import OffsetIOManager
from loss_utils import LossManager
from vlz_utils import vlz_pd_offset_img_both_sides,vlz_pd_offset_img,from_offset_img_to_rgb_img_both_sides
from train_utils import set_requires_grad,load_model,save_model

def write_cloth(batch,vt_offsets,out_dir,offset_io_manager,prefix):
    n_samples=len(vt_offsets)
    for i in range(n_samples):
        sample_id=batch['index'][i]
        sample_out_dir=join(out_dir,'{:08d}'.format(sample_id))
        if not isdir(sample_out_dir):
            os.makedirs(sample_out_dir)
        offset_io_manager.write_cloth_from_offsets(vt_offsets[i].detach().cpu().numpy(),sample_id,sample_out_dir,prefix=prefix)

def write_from_offset_imgs(offset_imgs,batch,managers,res_ctx,ctx,prefix):
    offset_manager=managers['offset_manager']
    if ctx['normalize_offset_imgs']:
        offset_imgs=offset_imgs*res_ctx['offset_img_std']+res_ctx['offset_img_mean']
    if ctx['use_uvn']:
        device=ctx['device']
        front_uvn_hats=batch['front_uvn_hat'].to(device)
        back_uvn_hats=batch['back_uvn_hat'].to(device)
        vt_offsets=offset_manager.get_offsets_from_uvn_offset_imgs_both_sides(offset_imgs,front_uvn_hats,back_uvn_hats)
    else:
        assert(False)
    write_cloth(batch,vt_offsets,out_dir=ctx['eval_out_dir'],offset_io_manager=managers['offset_io_manager'],prefix=prefix)

def normalize_rotations(rotations,res_ctx):
    return (rotations-res_ctx['rotation_mean'])/res_ctx['rotation_std']

def normalize_offset_imgs(offset_imgs,res_ctx):
    offset_imgs=(offset_imgs-res_ctx['offset_img_mean'])/res_ctx['offset_img_std']
    offset_imgs[:,:3,:,:]*=res_ctx['front_mask']
    offset_imgs[:,3:,:,:]*=res_ctx['back_mask']
    return offset_imgs

def tb_vlz(gt_offset_img,pd_offset_img,front_mask,back_mask,tb_logger,name,step):
    arr_stats={'minval':np.full(6,np.min(gt_offset_img)),'maxval':np.full(6,np.max(gt_offset_img))}
    vlz_img=vlz_pd_offset_img_both_sides(pd_offset_img,gt_offset_img,front_mask,back_mask,arr_stats=arr_stats)
    tb_logger.image_summary(name,np.expand_dims(vlz_img,axis=0),step)

def process_batch(batch,net,opt,managers,epoch,vlz=False,mode='train',res_ctx=None,ctx=None):
    device=ctx['device']
    dtype=ctx['dtype']
    rotations=batch['rotations'].to(device=device,dtype=dtype,non_blocking=True)
    gt_offset_imgs=batch['offset_img'].to(device=device,dtype=dtype,non_blocking=True)
    n_samples=len(rotations)

    if ctx['normalize_rotations']:
        rotations=normalize_rotations(rotations,res_ctx)
    if ctx['normalize_offset_imgs']:
        gt_offset_imgs=normalize_offset_imgs(gt_offset_imgs,res_ctx)

    pd_offset_imgs=net(rotations)

    front_masks=res_ctx['front_mask']
    back_masks=res_ctx['back_mask']

    loss_manager=managers['loss_manager']

    loss=loss_manager.get_pix_loss_both_sides(pd_offset_imgs,gt_offset_imgs,front_masks,back_masks)/n_samples
    loss_manager.add_item_loss('pix_loss',loss.item()*n_samples)

    loss_manager.add_total_loss(loss.item()*n_samples,n_samples)

    if mode=='eval' and ctx['write_pd']:
        write_from_offset_imgs(pd_offset_imgs,batch,managers,res_ctx,ctx,prefix='pd')
        write_cloth(batch,batch['vt_offset'],out_dir=ctx['eval_out_dir'],offset_io_manager=managers['offset_io_manager'],prefix='gt_cloth')
        return

    # step 
    if mode=='train':
        opt.zero_grad()
        loss.backward()
        opt.step()

    if vlz:
        gt_offset_imgs[:0,:3,:,:]*=front_masks
        gt_offset_imgs[:0,3:,:,:]*=back_masks
        tb_vlz(gt_offset_imgs[0].detach().cpu().numpy(),pd_offset_imgs[0].detach().cpu().numpy(),front_masks[0].squeeze().cpu().numpy(),back_masks[0].squeeze().cpu().numpy(),ctx['tb_logger'],'{}_img'.format(mode),epoch)


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
        if mode=='eval':
            managers['offset_manager']=OffsetManager(shared_data_dir=res_ctx['shared_data_dir'],ctx=ctx)
            managers['offset_io_manager']=OffsetIOManager(res_ctx=res_ctx,ctx=ctx)
        return managers

    managers=get_managers(ctx['res_ctx'])
    res_ctx=ctx['res_ctx']

    # process batches
    for idx,batch in enumerate(dataloader):
        vlz=idx==0 and mode=='train'
        process_batch(batch,net,opt,managers,epoch,vlz=vlz,mode=mode,res_ctx=res_ctx,ctx=ctx)

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

def load_model(model, optimizer, state):
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

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

def eval_model(dataloader,net,epoch,ctx):
    process_epoch(dataloader,net,opt=None,epoch=epoch,ctx=ctx,mode='eval')




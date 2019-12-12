######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import torch.nn.functional as F
import numpy as np
import os
from os.path import join,isdir
import time
from offset_img_utils import OffsetManager
from offset_io_utils import OffsetIOManager
from vlz_utils import vlz_pd_offset_img_both_sides,vlz_pd_offset_img,from_offset_img_to_rgb_img_both_sides
from gan_loss_utils import GANLossManager

def write_cloth(batch,vt_offsets,out_dir,offset_io_manager,prefix):
    n_samples=len(vt_offsets)
    for i in range(n_samples):
        sample_id=batch['index'][i]
        sample_out_dir=join(out_dir,'{:08d}'.format(sample_id))
        if not isdir(sample_out_dir):
            os.makedirs(sample_out_dir)
        offset_io_manager.write_cloth_from_offsets(vt_offsets[i].detach().cpu().numpy(),sample_id,sample_out_dir,prefix=prefix)

def write_from_offset_imgs(offset_imgs,batch,managers,level,res_ctx,ctx,prefix):
    offset_manager=managers['offset_manager']
    if level!=ctx['end_level']:
        offset_imgs=F.interpolate(offset_imgs,scale_factor=2**(ctx['end_level']-level),mode='bilinear')
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

def tb_vlz(gt_offset_img,pd_offset_img,front_mask,back_mask,tb_logger,name,step):
    arr_stats={'minval':np.full(6,np.min(gt_offset_img)),'maxval':np.full(6,np.max(gt_offset_img))}
    vlz_img=vlz_pd_offset_img_both_sides(pd_offset_img,gt_offset_img,front_mask,back_mask,arr_stats=arr_stats)
    tb_logger.image_summary(name,np.expand_dims(vlz_img,axis=0),step)

def normalize_offset_imgs(offset_imgs,res_ctx):
    offset_imgs=(offset_imgs-res_ctx['offset_img_mean'])/res_ctx['offset_img_std']
    offset_imgs[:,:3,:,:]*=res_ctx['front_mask']
    offset_imgs[:,3:,:,:]*=res_ctx['back_mask']
    return offset_imgs

def normalize_skin_imgs(skin_imgs,res_ctx):
    skin_imgs=(skin_imgs-res_ctx['skin_img_mean'])/res_ctx['skin_img_std']
    skin_imgs[:,:3,:,:]*=res_ctx['front_mask']
    skin_imgs[:,3:,:,:]*=res_ctx['back_mask']
    return skin_imgs

def permute(N):
    result=torch.randperm(N)
    preresult=result.clone()
    for i in range(N):
        if result[i]==i:
            next_i=(i+1)%N
            t=result[next_i].item()
            result[next_i]=result[i].item()
            result[i]=t # t must not be i
    return result

def cat_label(x,label):
    if label=='hero':
        hero_label=torch.tensor([1,0],device=x.device,dtype=x.dtype).view(1,-1)
        l=hero_label.repeat(x.size(0),1)
        return torch.cat([x,l],dim=1)
    elif label=='sim':
        sim_label=torch.tensor([0,1],device=x.device,dtype=x.dtype).view(1,-1)
        l=sim_label.repeat(x.size(0),1)
        return torch.cat([x,l],dim=1)
    else:
        print('label',label)
        assert(False)

def process_batch(batch,nets,opts,managers,epoch,level,alpha,mode='train',step_G=False,vlz=False,res_ctx=None,ctx=None):
    device=ctx['device']
    dtype=ctx['dtype']
    rotations=batch['rotations'].to(device=device,dtype=dtype,non_blocking=True)
    if ctx['normalize_rotations']:
        rotations=(rotations-res_ctx['rotation_mean'])/res_ctx['rotation_std']
    n_samples=len(rotations)
    if n_samples<4:
        return # robust batchnorm

    if ctx['use_style_gan']:
        if ctx['use_hero'] and ctx['cat_label']:
            rotations=cat_label(rotations,'hero') # hack
        rotations.requires_grad_(True)

    if ctx['use_style_gan'] and ctx['use_hero'] and ctx['use_label']:
        netG=lambda rot:nets['G'](rot,level,alpha,label='hero')
        netG_sim=lambda rot:nets['G'](rot,level,alpha,label='sim')
    else:
        netG=lambda rot:nets['G'](rot,level,alpha)
        netG_sim=netG

    if not ctx['pd_only']:
        gt_offset_imgs=batch['offset_img'].to(device=device,dtype=dtype,non_blocking=True)
        if ctx['normalize_offset_imgs']:
            gt_offset_imgs=normalize_offset_imgs(gt_offset_imgs,res_ctx)
        gt_offset_imgs=F.interpolate(gt_offset_imgs,scale_factor=2**(level-ctx['end_level']))

        if ctx['cat_skin_imgs'] and (mode=='train' or mode=='val'):
            skin_imgs=batch['skin_img'].to(device=device,dtype=dtype,non_blocking=True)
            if ctx['normalize_skin_imgs']:
                skin_imgs=normalize_skin_imgs(skin_imgs,res_ctx)
            skin_imgs=F.interpolate(skin_imgs,scale_factor=2**(level-ctx['end_level']))

    if ctx['use_hero'] and (mode=='train' or mode=='val'):
        sim_res_ctx=ctx['sim_res_ctx']
        sim_rotations=batch['sim']['rotations'].to(device=device,dtype=dtype,non_blocking=True)
        sim_gt_offset_imgs=batch['sim']['offset_img'].to(device=device,dtype=dtype,non_blocking=True)

        if ctx['normalize_rotations']:
            sim_rotations=(sim_rotations-sim_res_ctx['rotation_mean'])/sim_res_ctx['rotation_std']
        if ctx['normalize_offset_imgs']:
            sim_gt_offset_imgs=normalize_offset_imgs(sim_gt_offset_imgs,sim_res_ctx)
        sim_gt_offset_imgs=F.interpolate(sim_gt_offset_imgs,scale_factor=2**(level-ctx['end_level']))

        if ctx['cat_skin_imgs']:
            sim_skin_imgs=batch['sim']['skin_img'].to(device=device,dtype=dtype,non_blocking=True)
            if ctx['normalize_skin_imgs']:
                sim_skin_imgs=normalize_skin_imgs(sim_skin_imgs,res_ctx)
            sim_skin_imgs=F.interpolate(sim_skin_imgs,scale_factor=2**(level-ctx['end_level']))
        if ctx['use_style_gan']:
            if ctx['cat_label']:
                sim_rotations=cat_label(sim_rotations,'sim') # hack
            sim_rotations.requires_grad_(True)

    if step_G:
        pd_offset_imgs=netG(rotations)
    else:
        with torch.no_grad():
            pd_offset_imgs=netG(rotations)

    assert(not ctx['use_patches'])

    if mode=='eval' and ctx['write_pd']:
        write_from_offset_imgs(pd_offset_imgs,batch,managers,level,res_ctx,ctx,prefix='pd_{}'.format(level))
        if not ctx['pd_only']:
            write_from_offset_imgs(gt_offset_imgs,batch,managers,level,res_ctx,ctx,prefix='gt_{}'.format(level))
            write_cloth(batch,batch['vt_offset'],out_dir=ctx['eval_out_dir'],offset_io_manager=managers['offset_io_manager'],prefix='gt_cloth')
        if ctx['use_style_gan'] and ctx['use_hero']:
            # sim_rotations=cat_label(rotations_bak,'sim')
            sim_rotations=rotations
            if ctx['cat_label']:
                sim_rotations[:,-1]=1
                sim_rotations[:,-2]=0
            with torch.no_grad():
                sim_pd_offset_imgs=netG_sim(sim_rotations)
            write_from_offset_imgs(sim_pd_offset_imgs,batch,managers,level,res_ctx,ctx,prefix='pd_sim_{}'.format(level))
        return

    front_masks=res_ctx['level_to_front_mask'][level]
    back_masks=res_ctx['level_to_back_mask'][level]
    pd_offset_imgs[:,:3,:,:]*=front_masks
    pd_offset_imgs[:,3:,:,:]*=back_masks
    masks=torch.cat([front_masks.repeat(n_samples,3,1,1),back_masks.repeat(n_samples,3,1,1)],dim=1)
    if ctx['cat_skin_imgs']:
        masks=torch.cat([masks,torch.zeros_like(masks,dtype=dtype,device=device)],dim=1)
        # masks=torch.cat([masks,masks],dim=1)

    loss_manager=managers['loss_manager']
    real=torch.cat([gt_offset_imgs,skin_imgs],dim=1) if ctx['cat_skin_imgs'] else gt_offset_imgs
    fake=torch.cat([pd_offset_imgs,skin_imgs],dim=1) if ctx['cat_skin_imgs'] else pd_offset_imgs
    if ctx['use_pair_D']:
        assert(ctx['cat_skin_imgs'])
        perm_id=permute(n_samples).to(device=device,dtype=torch.long)
        perm=torch.cat([gt_offset_imgs[perm_id],skin_imgs],dim=1)

    real.requires_grad_(True)
    fake.requires_grad_(True)

    if ctx['use_hero']:
        if step_G:
            sim_pd_offset_imgs=netG_sim(sim_rotations)
        else:
            with torch.no_grad():
                sim_pd_offset_imgs=netG_sim(sim_rotations)

        sim_front_masks=sim_res_ctx['level_to_front_mask'][level]
        sim_back_masks=sim_res_ctx['level_to_back_mask'][level]
        sim_pd_offset_imgs[:,:3,:,:]*=sim_front_masks
        sim_pd_offset_imgs[:,3:,:,:]*=sim_back_masks
        sim_masks=torch.cat([sim_front_masks.repeat(n_samples,3,1,1),sim_back_masks.repeat(n_samples,3,1,1)],dim=1)
        if ctx['cat_skin_imgs']:
            sim_masks=torch.cat([sim_masks,torch.zeros_like(masks,dtype=dtype,device=device)],dim=1)

        sim_real=torch.cat([sim_gt_offset_imgs,sim_skin_imgs],dim=1) if ctx['cat_skin_imgs'] else sim_gt_offset_imgs
        sim_fake=torch.cat([sim_pd_offset_imgs,sim_skin_imgs],dim=1) if ctx['cat_skin_imgs'] else sim_pd_offset_imgs
        sim_real.requires_grad_(True)
        sim_fake.requires_grad_(True)

    if ctx['use_ctgr_D'] or ctx['use_ctgr_D_v3']:
        if ctx['use_ctgr_D']:
            netD=lambda offset_imgs:nets['D'](offset_imgs,level,alpha).squeeze()[:,0]
            netD_sim=lambda offset_imgs:nets['D'](offset_imgs,level,alpha).squeeze()[:,1]
        elif ctx['use_ctgr_D_v3']:
            netD=lambda offset_imgs:nets['D'](offset_imgs,level,alpha)
            netD_sim=netD
        loss_D_fake=loss_manager.get_loss_D(netD,fake)
        loss_manager.add_item_loss('loss_D_fake',loss_D_fake.item()*n_samples,n_samples)
        loss_D_real=loss_manager.get_loss_D(netD,real)
        loss_manager.add_item_loss('loss_D_real',loss_D_real.item()*n_samples,n_samples)
        loss_D_hero=loss_D_fake-loss_D_real
        loss_manager.add_item_loss('loss_diff_hero',loss_D_hero.item()*n_samples,n_samples)

        loss_D_sim_fake=loss_manager.get_loss_D(netD_sim,sim_fake)
        loss_manager.add_item_loss('loss_D_sim_fake',loss_D_sim_fake.item()*n_samples,n_samples)
        loss_D_sim_real=loss_manager.get_loss_D(netD_sim,sim_real)
        loss_manager.add_item_loss('loss_D_sim_real',loss_D_sim_real.item()*n_samples,n_samples)
        loss_D_sim=loss_D_sim_fake-loss_D_sim_real
        loss_manager.add_item_loss('loss_diff_sim',loss_D_sim.item()*n_samples,n_samples)

        loss_D=loss_D_hero+loss_D_sim*ctx['lambda_ctgr_sim']

        if ctx['lambda_R1']!=0:
            loss_R1=loss_manager.get_loss_R1(netD,real,masks)
            loss_manager.add_item_loss('loss_R1',loss_R1.item()*n_samples,n_samples)
            loss_D+=ctx['lambda_R1']*loss_R1

            loss_R1_sim=loss_manager.get_loss_R1(netD_sim,sim_real,masks)
            loss_manager.add_item_loss('loss_R1_sim',loss_R1_sim.item()*n_samples,n_samples)
            loss_D+=ctx['lambda_R1']*loss_R1_sim*ctx['lambda_ctgr_sim']

    elif ctx['use_style_gan']:
        if ctx['use_label']:
            netD=lambda offset_imgs,cond:nets['D'](offset_imgs,cond,level,alpha,label='hero')
            netD_sim=lambda offset_imgs,cond:nets['D'](offset_imgs,cond,level,alpha,label='sim')
        else:
            netD=lambda offset_imgs,cond:nets['D'](offset_imgs,cond,level,alpha)
            netD_sim=netD
        if ctx['use_proj']:
            loss_D_fake,loss_D_fake_cond=loss_manager.get_proj_loss_D(netD,fake,rotations)
            loss_manager.add_item_loss('loss_D_fake',loss_D_fake.item()*n_samples,n_samples)
            loss_manager.add_item_loss('loss_D_fake_cond',loss_D_fake_cond.item()*n_samples,n_samples)
            loss_D_real,loss_D_real_cond=loss_manager.get_proj_loss_D(netD,real,rotations)
            loss_manager.add_item_loss('loss_D_real',loss_D_real.item()*n_samples,n_samples)
            loss_manager.add_item_loss('loss_D_real_cond',loss_D_real_cond.item()*n_samples,n_samples)
            loss_D_hero=loss_D_fake-loss_D_real+(loss_D_fake_cond-loss_D_real_cond)*ctx['lambda_proj']
            loss_manager.add_item_loss('loss_D_hero',loss_D_hero.item()*n_samples,n_samples)
            loss_D=loss_D_hero

            if ctx['use_hero']:
                loss_D_sim_fake,loss_D_sim_fake_cond=loss_manager.get_proj_loss_D(netD_sim,sim_fake,sim_rotations)
                loss_manager.add_item_loss('loss_D_sim_fake',loss_D_sim_fake.item()*n_samples,n_samples)
                loss_manager.add_item_loss('loss_D_sim_fake_cond',loss_D_sim_fake_cond.item()*n_samples,n_samples)
                loss_D_sim_real,loss_D_sim_real_cond=loss_manager.get_proj_loss_D(netD_sim,sim_real,sim_rotations)
                loss_manager.add_item_loss('loss_D_sim_real',loss_D_sim_real.item()*n_samples,n_samples)
                loss_manager.add_item_loss('loss_D_sim_real_cond',loss_D_sim_real_cond.item()*n_samples,n_samples)
                loss_D_sim=loss_D_sim_fake-loss_D_sim_real+(loss_D_sim_fake_cond-loss_D_sim_real_cond)*ctx['lambda_cond']
                loss_manager.add_item_loss('loss_D_sim',loss_D_sim.item()*n_samples,n_samples)
                loss_D+=loss_D_sim

            if ctx['lambda_R1']!=0:
                loss_R1=loss_manager.get_proj_loss_R1(loss_D_real+loss_D_real_cond,real,rotations,masks)
                loss_manager.add_item_loss('loss_R1',loss_R1.item()*n_samples,n_samples)
                loss_D+=ctx['lambda_R1']*loss_R1

                if ctx['use_hero']:
                    loss_R1_sim=loss_manager.get_proj_loss_R1(loss_D_sim_real+loss_D_sim_real_cond,sim_real,sim_rotations,masks)
                    loss_manager.add_item_loss('loss_R1_sim',loss_R1_sim.item()*n_samples,n_samples)
                    loss_D+=ctx['lambda_R1']*loss_R1_sim
        else:
            loss_D_fake=loss_manager.get_cond_loss_D(netD,fake,rotations)
            loss_manager.add_item_loss('loss_D_fake',loss_D_fake.item()*n_samples,n_samples)
            loss_D_real=loss_manager.get_cond_loss_D(netD,real,rotations)
            loss_manager.add_item_loss('loss_D_real',loss_D_real.item()*n_samples,n_samples)
            loss_D_hero=loss_D_fake-loss_D_real
            loss_manager.add_item_loss('loss_D_hero',loss_D_hero.item()*n_samples,n_samples)
            loss_D=loss_D_hero

            if ctx['use_hero']:
                loss_D_sim_fake=loss_manager.get_cond_loss_D(netD_sim,sim_fake,sim_rotations)
                loss_manager.add_item_loss('loss_D_sim_fake',loss_D_sim_fake.item()*n_samples,n_samples)
                loss_D_sim_real=loss_manager.get_cond_loss_D(netD_sim,sim_real,sim_rotations)
                loss_manager.add_item_loss('loss_D_sim_real',loss_D_sim_real.item()*n_samples,n_samples)
                loss_D_sim=loss_D_sim_fake-loss_D_sim_real
                loss_manager.add_item_loss('loss_D_sim',loss_D_sim.item()*n_samples,n_samples)
                loss_D+=loss_D_sim*ctx['lambda_ctgr_sim']

            if ctx['lambda_R1']!=0:
                loss_R1=loss_manager.get_cond_loss_R1(netD,real,rotations,masks)
                loss_manager.add_item_loss('loss_R1',loss_R1.item()*n_samples,n_samples)
                loss_D+=ctx['lambda_R1']*loss_R1

                if ctx['use_hero']:
                    loss_R1_sim=loss_manager.get_cond_loss_R1(netD_sim,sim_real,sim_rotations,sim_masks)
                    loss_manager.add_item_loss('loss_R1_sim',loss_R1_sim.item()*n_samples,n_samples)
                    loss_D+=ctx['lambda_R1']*ctx['lambda_ctgr_sim']*loss_R1_sim

            if ctx['lambda_R2']!=0:
                loss_R2=loss_manager.get_cond_loss_R2(netD,fake,rotations,masks)
                loss_manager.add_item_loss('loss_R2',loss_R2.item()*n_samples,n_samples)
                loss_D+=ctx['lambda_R2']*loss_R2

                if ctx['use_hero']:
                    loss_R2_sim=loss_manager.get_cond_loss_R2(netD_sim,sim_fake,sim_rotations,sim_masks)
                    loss_manager.add_item_loss('loss_R2_sim',loss_R2_sim.item()*n_samples,n_samples)
                    loss_D+=ctx['lambda_R2']*ctx['lambda_ctgr_sim']*loss_R2_sim

            if ctx['lambda_zgp']!=0:
                loss_zgp=loss_manager.get_cond_loss_zgp(netD,real,fake,rotations,masks)
                loss_manager.add_item_loss('loss_zgp',loss_zgp.item()*n_samples,n_samples)
                loss_D+=ctx['lambda_zgp']*loss_zgp

                if ctx['use_hero']:
                    loss_zgp_sim=loss_manager.get_cond_loss_zgp(netD_sim,sim_real,sim_fake,sim_rotations,sim_masks)
                    loss_manager.add_item_loss('loss_zgp_sim',loss_zgp_sim.item()*n_samples,n_samples)
                    loss_D+=ctx['lambda_zgp']*ctx['lambda_ctgr_sim']*loss_zgp_sim

        if ctx['lambda_consensus']!=0:
            loss_consensus_D=loss_manager.get_consensus_loss(nets['D'],loss_D)
            loss_manager.add_item_loss('loss_consensus_D',loss_consensus_D.item()*n_samples,n_samples)
            loss_D+=loss_consensus_D*ctx['lambda_consensus']

    elif ctx['use_ctgr_D_v2']: # risk saturating and smaller gradients
        netD_raw=lambda offset_imgs:nets['D'](offset_imgs,level,alpha).squeeze()
        raw_real=netD_raw(real)
        raw_fake=netD_raw(fake)
        raw_sim_real=netD_raw(sim_real)
        raw_sim_fake=netD_raw(sim_fake)
        loss_D_real=loss_manager.get_cat_loss_D(raw_real,'real')
        loss_manager.add_item_loss('loss_D_real',loss_D_real.item()*n_samples,n_samples)
        loss_D_fake=loss_manager.get_cat_loss_D(raw_fake,'fake')
        loss_manager.add_item_loss('loss_D_fake',loss_D_fake.item()*n_samples,n_samples)
        loss_D_sim=loss_manager.get_cat_loss_D(raw_sim_real,'sim')
        loss_manager.add_item_loss('loss_D_sim',loss_D_sim.item()*n_samples,n_samples)
        loss_D_sim_fake=loss_manager.get_cat_loss_D(raw_sim_fake,'fake')
        loss_manager.add_item_loss('loss_D_sim_fake',loss_D_sim_fake.item()*n_samples,n_samples)
        loss_D=loss_D_real+loss_D_sim+(loss_D_fake+loss_D_sim_fake)/2 # might not need to be divided by 2, just want the three labels to be more balanced

        if ctx['lambda_R1']!=0:
            loss_R1_real=loss_manager.get_loss_R1(netD=None,x=real,y=loss_D_real,masks=masks)
            loss_manager.add_item_loss('loss_R1_real',loss_R1_real.item()*n_samples,n_samples)
            loss_R1_sim=loss_manager.get_loss_R1(netD=None,x=sim_real,y=loss_D_sim,masks=masks)
            loss_manager.add_item_loss('loss_R1_sim',loss_R1_sim.item()*n_samples,n_samples)
            loss_R1=loss_R1_real+loss_R1_sim
            loss_D+=loss_R1*ctx['lambda_R1']*loss_R1

    elif ctx['use_lsgan']:
        netD=lambda offset_imgs:nets['D'](offset_imgs,level,alpha)
        netD_sim=netD

        loss_D_fake=loss_manager.get_sqr_loss(netD,fake,tgt=0)
        loss_manager.add_item_loss('loss_D_fake',loss_D_fake.item()*n_samples,n_samples)
        loss_D_real=loss_manager.get_sqr_loss(netD,real,tgt=1)
        loss_manager.add_item_loss('loss_D_real',loss_D_real.item()*n_samples,n_samples)
        loss_D_hero=loss_D_fake+loss_D_real
        loss_manager.add_item_loss('loss_D_hero',loss_D_hero.item()*n_samples,n_samples)

        loss_D_sim_fake=loss_manager.get_sqr_loss(netD_sim,sim_fake,tgt=0)
        loss_manager.add_item_loss('loss_D_sim_fake',loss_D_sim_fake.item()*n_samples,n_samples)
        loss_D_sim_real=loss_manager.get_sqr_loss(netD_sim,sim_real,tgt=ctx['lambda_ctgr_sim'])
        loss_manager.add_item_loss('loss_D_sim_real',loss_D_sim_real.item()*n_samples,n_samples)
        loss_D_sim=loss_D_sim_fake+loss_D_sim_real
        loss_manager.add_item_loss('loss_D_sim',loss_D_sim.item()*n_samples,n_samples)

        loss_D=loss_D_hero+loss_D_sim

    else:
        netD=lambda offset_imgs:nets['D'](offset_imgs,level,alpha)
        loss_D_fake=loss_manager.get_loss_D(netD,fake)
        loss_manager.add_item_loss('loss_D_fake',loss_D_fake.item()*n_samples,n_samples)
        loss_D_real=loss_manager.get_loss_D(netD,real)
        loss_manager.add_item_loss('loss_D_real',loss_D_real.item()*n_samples,n_samples)
        loss_D=loss_D_fake-loss_D_real
        loss_manager.add_item_loss('loss_diff',loss_D.item()*n_samples,n_samples)
        if ctx['use_pair_D']:
            loss_D_perm=loss_manager.get_loss_D(netD,perm)
            loss_manager.add_item_loss('loss_D_perm',loss_D_perm.item()*n_samples,n_samples)
            loss_D+=loss_D_perm-loss_D_real

        if ctx['lambda_gp']!=0:
            loss_gp=loss_manager.get_loss_gp(netD,fake,real,masks)
            loss_manager.add_item_loss('loss_gp',loss_gp.item()*n_samples,n_samples)
            loss_D+=ctx['lambda_gp']*loss_gp
        if ctx['lambda_R1']!=0:
            loss_R1=loss_manager.get_loss_R1(netD,real,masks)
            loss_manager.add_item_loss('loss_R1',loss_R1.item()*n_samples,n_samples)
            loss_D+=ctx['lambda_R1']*loss_R1

    if mode=='train':
        opts['D'].zero_grad()
        loss_D.backward(retain_graph=True)
        opts['D'].step()

        if step_G:
            if ctx['use_ctgr_D'] or ctx['use_ctgr_D_v3']:
                # this is wrong because of the inconsistency
                # loss_G_hero=loss_manager.get_loss_G(netD,fake)+loss_manager.get_loss_G(netD,sim_fake)
                # loss_G_sim=loss_manager.get_loss_G(netD_sim,sim_fake)+loss_manager.get_loss_G(netD_sim,fake)
                loss_G_hero=loss_manager.get_loss_G(netD,fake)
                loss_manager.add_item_loss('loss_G_hero',loss_G_hero.item()*n_samples,n_samples)
                loss_G_sim=loss_manager.get_loss_G(netD_sim,sim_fake)
                loss_manager.add_item_loss('loss_G_sim',loss_G_sim.item()*n_samples,n_samples)
                loss_G=loss_G_hero+loss_G_sim*ctx['lambda_ctgr_sim']
            elif ctx['use_style_gan']:
                if ctx['use_proj']:
                    loss_G_fake,loss_G_cond=loss_manager.get_proj_loss_G(netD,fake,rotations)
                    loss_G_hero=loss_G_fake+loss_G_cond*ctx['lambda_cond']
                    loss_manager.add_item_loss('loss_G_hero',loss_G_hero.item()*n_samples,n_samples)
                    loss_G=loss_G_hero
                    if ctx['use_hero']:
                        loss_G_sim_fake,loss_G_sim_cond=loss_manager.get_proj_loss_G(netD_sim,sim_fake,sim_rotations)
                        loss_G_sim=loss_G_sim_fake+loss_G_sim_cond*ctx['lambda_cond']
                        loss_manager.add_item_loss('loss_G_sim',loss_G_sim.item()*n_samples,n_samples)
                        loss_G+=loss_G_sim*ctx['lambda_ctgr_sim']
                else:
                    loss_G_hero=loss_manager.get_cond_loss_G(netD,fake,rotations)
                    loss_manager.add_item_loss('loss_G_hero',loss_G_hero.item()*n_samples,n_samples)
                    loss_G=loss_G_hero
                    # loss_G=0
                    if ctx['use_hero']:
                        loss_G_sim=loss_manager.get_cond_loss_G(netD_sim,sim_fake,sim_rotations)
                        loss_manager.add_item_loss('loss_G_sim',loss_G_sim.item()*n_samples,n_samples)
                        loss_G+=loss_G_sim*ctx['lambda_ctgr_sim']

            elif ctx['use_ctgr_D_v2']:
                # this is wrong, I should use the updated netD
                loss_G_fake1=loss_manager.get_cat_loss_G(raw_fake,lambda_ctgr_sim=ctx['lambda_ctgr_sim'])
                loss_G_fake2=loss_manager.get_cat_loss_G(raw_sim_fake,lambda_ctgr_sim=ctx['lambda_ctgr_sim'])
                loss_G=loss_G_fake1+loss_G_fake2
            elif ctx['use_lsgan']:
                loss_G_hero=loss_manager.get_sqr_loss(netD,fake,tgt=1)
                loss_G_sim=loss_manager.get_sqr_loss(netD,sim_fake,tgt=ctx['lambda_ctgr_sim'])
                # maybe try this later
                # loss_G_sim=loss_manager.get_loss_G(netD,sim_fake,tgt=ctx['lambda_ctgr_sim'])
                loss_G=loss_G_hero+loss_G_sim
            else:
                loss_G=loss_manager.get_loss_G(netD,fake)
                loss_manager.add_item_loss('loss_G',loss_G.item()*n_samples,n_samples)

            if ctx['lambda_l1']!=0:
                loss_l1=loss_manager.get_loss_l1(pd_offset_imgs,gt_offset_imgs,masks[:,:6])
                loss_manager.add_item_loss('loss_l1',loss_l1.item()*n_samples,n_samples)
                loss_G+=ctx['lambda_l1']*loss_l1

            if ctx['lambda_sim_l1']!=0:
                assert(ctx['use_hero'])
                if not (ctx['use_ctgr_D'] or ctx['use_ctgr_D_v2'] or ctx['use_ctgr_D_v3']): # already has sim_pd_offset_imgs
                    sim_pd_offset_imgs=nets['G'](sim_rotations,level,alpha)
                loss_sim_l1=loss_manager.get_loss_l1(sim_pd_offset_imgs,sim_gt_offset_imgs,masks[:,:6])
                loss_manager.add_item_loss('loss_sim_l1',loss_sim_l1.item()*n_samples,n_samples)
                loss_G+=ctx['lambda_sim_l1']*loss_sim_l1

            if ctx['lambda_consensus']!=0:
                loss_consensus_G=loss_manager.get_consensus_loss(nets['G'],loss_G)
                loss_manager.add_item_loss('loss_consensus_G',loss_consensus_G.item()*n_samples,n_samples)
                loss_G+=loss_consensus_G*ctx['lambda_consensus']

            opts['G'].zero_grad()
            loss_G.backward()
            opts['G'].step()
    if vlz:
        tb_vlz(gt_offset_imgs[0].detach().cpu().numpy(),pd_offset_imgs[0].detach().cpu().numpy(),front_masks[0].squeeze().cpu().numpy(),back_masks[0].squeeze().cpu().numpy(),ctx['tb_logger'],'{}_img'.format(mode),epoch)

        if ctx['use_hero'] and mode=='train':
            tb_vlz(sim_gt_offset_imgs[0].detach().cpu().numpy(),sim_pd_offset_imgs[0].detach().cpu().numpy(),sim_front_masks[0].squeeze().cpu().numpy(),sim_back_masks[0].squeeze().cpu().numpy(),ctx['tb_logger'],'{}_sim_img'.format(mode),epoch)

def process_epoch(dataloader,nets,opts,epoch,ctx,mode='train'):
    epoch_start_time=time.time()
    if mode=='train':
        for key,net in nets.items():
            net.train()
        torch.set_grad_enabled(True)
    else: # validate or evaluation
        for key,net in nets.items():
            net.eval()
        if mode!='val':
            torch.set_grad_enabled(False)

    res_ctx=ctx['res_ctx']
    epochs_per_level=ctx['epochs_per_level']
    level=ctx['start_level']+epoch//epochs_per_level
    alpha=epoch%ctx['epochs_per_level']/ctx['epochs_per_level']*2
    if level>ctx['end_level']:
        level=ctx['end_level']
        alpha=2
    alpha_step=2/ctx['epochs_per_level']/len(dataloader)
    # print('epoch',epoch,'level',level,'start alpha {:.6f}'.format(alpha),'alpha_step {:.6f}'.format(alpha_step),'n_batches',len(dataloader))
    
    # managers and res_ctx
    def get_managers(res_ctx):
        managers={
            # 'offset_manager':OffsetManager(shared_data_dir=res_ctx['shared_data_dir'],ctx=ctx),
            'loss_manager':GANLossManager(ctx=ctx)
        }
        if mode=='eval':
            managers['offset_manager']=OffsetManager(shared_data_dir=res_ctx['shared_data_dir'],ctx=ctx)
            managers['offset_io_manager']=OffsetIOManager(res_ctx=res_ctx,ctx=ctx)
        return managers

    managers=get_managers(ctx['res_ctx'])

    process_batch_time=0

    if (mode=='train' or mode=='val') and ctx['use_hero']:
        dataloader.dataset.reset_sim_ids()

    # process batches
    for idx,batch in enumerate(dataloader):

        vlz=idx==0 and (mode=='train' or mode=='val') and epoch%ctx['vlz_per_epoch']==0
        # with torch.autograd.detect_anomaly():
        #     process_batch(batch,nets,opts,managers,epoch,level,alpha+alpha_step*idx,step_G=(idx+1)%ctx['n_critic']==0,mode=mode,vlz=vlz,res_ctx=res_ctx,ctx=ctx)
        batch_start_time=time.time()
        process_batch(batch,nets,opts,managers,epoch,level,alpha+alpha_step*idx,step_G=(idx+1)%ctx['n_critic']==0,mode=mode,vlz=vlz,res_ctx=res_ctx,ctx=ctx)
        batch_end_time=time.time()
        process_batch_time+=batch_end_time-batch_start_time

    # log loss
    if mode=='train' or mode=='val':
        def log_loss(loss_manager,tb_logger,surfix=''):
            item_loss=loss_manager.get_item_loss()
            for name,loss in item_loss.items():
                tb_logger.scalar_summary('{}/{}{}'.format(mode,name,surfix),loss,epoch)

        log_loss(managers['loss_manager'],ctx['tb_logger'])

    # print necessary information
    epoch_end_time=time.time()

    if epoch%ctx['print_per_epoch']==0:
        print(mode,'epoch:',epoch,', time:',time.strftime('%H:%M:%S',time.gmtime(epoch_end_time-epoch_start_time)),',batch_time:',time.strftime('%H:%M:%S',time.gmtime(process_batch_time)),','.join(['{}:{:.6f}'.format(loss_name,loss_value) for loss_name,loss_value in managers['loss_manager'].get_item_loss().items()]))


def save_model(nets, opts, lr, epoch, save_path):
    state = {
            'epoch': epoch + 1,
            'netD': nets['D'].state_dict(),
            'netG': nets['G'].state_dict(),
            'optD': opts['D'].state_dict(),
            'optG': opts['G'].state_dict(),
            'lr': lr,
            }
    torch.save(state, save_path)

def load_model(nets, opts, state):
    nets['D'].load_state_dict(state['netD'])
    nets['G'].load_state_dict(state['netG'])
    opts['D'].load_state_dict(state['optD'])
    opts['G'].load_state_dict(state['optG'])


def train_model(dataloaders,nets,opts,start_epoch,ctx):

    rundir=ctx['rundir']
    with open(join(rundir,'train_args.txt'),'w') as fout:
        for key,value in ctx.items():
            if isinstance(value,str) or isinstance(value,float) or isinstance(value,int) or isinstance(value,bool) or value is None:
                fout.write('{}:{}\n'.format(key,value))

    save_dir=join(rundir,'saved_models')
    if not isdir(save_dir):
        os.makedirs(save_dir)

    save_every_epoch=ctx.get('save_every_epoch',10)
    num_epochs=ctx['num_epochs']
    print('start training from epoch',start_epoch,'auto save every',save_every_epoch,'epochs')
    
    for epoch in range(start_epoch,num_epochs):
        if epoch<(ctx['end_level']-ctx['start_level']+1)*ctx['epochs_per_level'] and epoch%ctx['epochs_per_level']==0:
            print('reset opts state')
            opts=ctx['init_opts'](nets)

        process_epoch(dataloaders['train'],nets,opts,epoch,ctx,mode='train')

        # with torch.cuda.profiler.profile():
        #     with torch.autograd.profiler.emit_nvtx():
        #         process_epoch(dataloaders['train'],nets,opts,epoch,ctx,mode='train')

        # with torch.autograd.profiler.profile() as prof:
        #     process_epoch(dataloaders['train'],nets,opts,epoch,ctx,mode='train')
        # print(prof)
        # prof.export_chrome_trace('trace.html')

        process_epoch(dataloaders['val'],nets,None,epoch,ctx,mode='val')

        if (epoch+1)%save_every_epoch==0:
            save_path=join(save_dir,'checkpoint-{}.pth.tar'.format(epoch+1))
            save_model(nets,opts,ctx['lr'],epoch,save_path)

def eval_model(dataloader,nets,epoch,ctx):
    process_epoch(dataloader,nets,opts=None,epoch=epoch,ctx=ctx,mode='eval')
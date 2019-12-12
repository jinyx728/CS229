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
from vlz_utils import vlz_pd_offset_img_both_sides,vlz_pd_offset_img
from loss_utils import LossManager
import gzip
from img_sample_func import ImgSampleBothSidesModule,ImgSample2ndOrderBothSidesModule
from obj_io import Obj,write_obj

def set_requires_grad(layers, requires_grad):
    for param in layers.parameters():
        param.requires_grad = requires_grad

def count_parameters(model):
    count = 0
    for p in model.parameters():
        if p.requires_grad:
            count += np.prod(p.size())
    return count

def print_layer_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print('layer', name, 'numel', param.numel())

def write_cloth(batch,vt_offsets,out_dir,offset_io_manager,prefix):
    n_samples=len(vt_offsets)
    for i in range(n_samples):
        sample_id=batch['index'][i]
        sample_out_dir=join(out_dir,'{:08d}'.format(sample_id))
        if not isdir(sample_out_dir):
            os.makedirs(sample_out_dir)
        offset_io_manager.write_cloth_from_offsets(vt_offsets[i].detach().cpu().numpy(),sample_id,sample_out_dir,prefix=prefix)

def write_objs(batch,vts,fc,out_dir,prefix):
    n_samples=len(vts)
    for i in range(n_samples):
        sample_id=batch['index'][i]
        sample_out_dir=join(out_dir,'{:08d}'.format(sample_id))
        if not isdir(sample_out_dir):
            os.makedirs(sample_out_dir)
        write_obj(Obj(v=vts[i].detach().cpu().numpy(),f=fc),join(sample_out_dir,'{}_{:08d}.obj'.format(prefix,sample_id)))

def write_npys(batch,vt_offsets,out_dir,prefix):
    n_samples=len(vt_offsets)
    for i in range(n_samples):
        sample_id=batch['index'][i]
        sample_out_dir=join(out_dir,'{:08d}'.format(sample_id))
        if not isdir(sample_out_dir):
            os.makedirs(sample_out_dir)
        np.save(join(sample_out_dir,'{}_{:08d}.npy'.format(prefix,sample_id)),vt_offsets[i].detach().cpu().numpy())

def write_offsets(batch,vt_offsets,out_dir,prefix):
    n_samples=len(vt_offsets)
    if not isdir(out_dir):
        os.makedirs(out_dir)
    for i in range(n_samples):
        sample_id=batch['index'][i]
        np.save(join(out_dir,'{}_{:08d}.npy'.format(prefix,sample_id)),vt_offsets[i].detach().cpu().numpy())

# def write_output(batch,pd_vt_offsets,out_dir,offset_io_manager):
#     n_samples=len(pd_vt_offsets)
#     gt_vt_offsets=batch['vt_offset']
#     for i in range(n_samples):
#         sample_id=batch['index'][i]
#         sample_out_dir=join(out_dir,'{:08d}'.format(sample_id))
#         if not exists(sample_out_dir):
#             os.makedirs(sample_out_dir)
#         offset_io_manager.write_cloth_from_offsets(pd_vt_offsets[i].cpu().numpy(),sample_id,sample_out_dir,prefix='pd_cloth')
#         offset_io_manager.write_cloth_from_offsets(gt_vt_offsets[i].cpu().numpy(),sample_id,sample_out_dir,prefix='gt_cloth')

def write_imgs(batch,offset_imgs,out_dir):
    n_samples=len(offset_imgs)
    for i in range(n_samples):
        sample_id=batch['index'][i]
        sample_out_dir=join(out_dir,'{:08d}'.format(sample_id))
        if not exists(sample_out_dir):
            os.makedirs(sample_out_dir)
        with gzip.open(join(sample_out_dir,'pd_img_{:08d}.npy.gz'.format(sample_id)),'wb') as f:
            np.save(file=f,arr=offset_imgs[i].permute(1,2,0).cpu().numpy())

def write_diff_imgs(batch,diff_imgs,front_masks,back_masks,out_dir):
    diff_imgs[:,:3,:,:]*=front_masks
    diff_imgs[:,3:,:,:]*=back_masks
    n_samples=len(diff_imgs)
    for i in range(n_samples):
        sample_id=batch['index'][i]
        out_path=join(out_dir,'diff_img_{:08d}.npy.gz'.format(sample_id))
        with gzip.open(out_path,'wb') as f:
            np.save(file=f,arr=diff_imgs[i].permute(1,2,0).cpu().numpy())

def process_batch(batch,net,opt,managers,epoch,mode='train',vlz=False,res_ctx=None,ctx=None):
    device=ctx['device']
    dtype=ctx['dtype']
    rotations=batch['rotations'].to(device)
    if ctx['use_diff']:
        gt_offset_imgs=batch['diff_img'].to(device)
    elif not ctx['pd_only']:
        gt_offset_imgs=batch['offset_img'].to(device)
    n_samples=len(rotations)

    # start_time=time.time()
    pd_offset_imgs=net(rotations)
    if ctx['use_variable_m']:
        m_net=ctx['m_net']
        m_opt=ctx['m_opt']
        pd_m_imgs=m_net(rotations)

    offset_manager=managers['offset_manager']
    loss_manager=managers['loss_manager']

    if ctx['use_patches']:
        crop_masks=batch['crop_mask'].to(device)
        if ctx['use_ecos'] or ctx['use_spring'] or ctx['use_cudaqs'] or ctx['use_vt_loss'] or mode=='eval':
            vts_in_crop=res_ctx['vts_in_crop']
            vt_ids_in_crop=res_ctx['vt_ids_in_crop']
            if ctx['use_uvn']:
                uvn_hats=batch['uvn_hat'].to(device)
                pd_vt_offsets=offset_manager.get_offsets_from_uvn_offset_imgs(pd_offset_imgs,vts_in_crop,uvn_hats)
            else:
                pd_vt_offsets=offset_manager.get_offsets_from_offset_imgs(pd_offset_imgs,vts_in_crop)
            gt_vt_offsets=batch['vt_offset'].to(device,dtype=dtype)
        if ctx['use_ecos'] or ctx['use_spring'] or ctx['use_cudaqs'] or mode=='eval':
            if ctx['use_variable_m']:
                pd_ms=offset_manager.get_offsets_from_offset_imgs(pd_m_imgs,vts_in_crop).squeeze(2)
                pd_ms=torch.clamp(pd_ms*0.01+1,1e-6)
            skin=batch['skin'].to(device,dtype=dtype)
            pd_vts=skin+pd_vt_offsets
    else:
        if ctx['use_diff']:
            full_gt_offset_imgs=batch['offset_img'].to(device)
            tmp_pd_offset_imgs=pd_offset_imgs
            pd_offset_imgs=full_gt_offset_imgs-gt_offset_imgs+pd_offset_imgs

        front_masks=res_ctx['front_mask']
        back_masks=res_ctx['back_mask']

        if ctx['use_diff']:
            pd_offset_imgs=tmp_pd_offset_imgs # set back

        if ctx['use_cvxpy'] or ctx['calc_vt_loss']:
            img_sample_module=managers['img_sample_module']
            gt_vt_offsets=batch['vt_offset'].to(device,dtype=dtype)
            pd_vt_offsets=img_sample_module(pd_offset_imgs)
        if ctx['use_ecos'] or ctx['use_spring'] or ctx['use_cudaqs'] or ctx['use_vt_loss'] or (mode=='eval' and not ctx['use_gen_diff']):
            if ctx['use_variable_m']:
                pd_ms=offset_manager.get_offsets_from_offset_imgs_both_sides(pd_m_imgs).squeeze(2)
                pd_ms=torch.clamp(pd_ms*0.01+1,1e-6)
            if not ctx['pd_only']:
                gt_vt_offsets=batch['vt_offset'].to(device,dtype=dtype)
            if ctx['use_uvn']:
                front_uvn_hats=batch['front_uvn_hat'].to(device)
                back_uvn_hats=batch['back_uvn_hat'].to(device)
                pd_vt_offsets=offset_manager.get_offsets_from_uvn_offset_imgs_both_sides(pd_offset_imgs,front_uvn_hats,back_uvn_hats)
            else:
                pd_vt_offsets=offset_manager.get_offsets_from_offset_imgs_both_sides(pd_offset_imgs)
            pd_vt_offsets.type(dtype)
        if ctx['use_cvxpy'] or ctx['use_ecos'] or ctx['use_spring'] or ctx['use_cudaqs']:
            skin=batch['skin'].to(device,dtype=dtype)
            pd_vts=skin+pd_vt_offsets

    # end_time=time.time()
    # print('eval time',end_time-start_time)

    # loss computation
    loss=0

    if ctx['use_cvxpy'] or ctx['use_ecos'] or ctx['use_cudaqs']:
        if ctx['use_cvxpy']:
            proj_module=ctx['cvx_module']
        if ctx['use_ecos']:
            proj_module=ctx['ecos_module']
        if ctx['use_cudaqs']:
            proj_module=ctx['cudaqs_module']
        if not ctx['pd_only']:
            pre_proj_loss=loss_manager.get_mse_loss(pd_vt_offsets,gt_vt_offsets,norm_vts=False)
            loss_manager.add_item_loss('pre_proj_loss',pre_proj_loss.item()*n_samples)
        if ctx['use_variable_m']:
            cr_vts=proj_module(pd_vts,pd_ms)
        else:
            cr_vts=proj_module(pd_vts)
        cr_vt_offsets=cr_vts-skin
        if not ctx['pd_only']:
            post_proj_loss=loss_manager.get_mse_loss(cr_vt_offsets,gt_vt_offsets,norm_vts=False)

        if ctx['use_avg_loss']:
            if pre_proj_loss.item()>1:
                print('unusual pre_proj_loss:',pre_proj_loss.item())
            if post_proj_loss.item()>1:
                print('unusual post_proj_loss:',post_proj_loss.item())
            loss+=(pre_proj_loss+post_proj_loss)/2
            # loss+=pre_proj_loss
        else:
            loss+=post_proj_loss
        loss_manager.add_item_loss('post_proj_loss',post_proj_loss.item()*n_samples)
    elif ctx['use_spring']:
        proj_module=ctx['spring_module']
        cr_vts_list=[pd_vts]
        x=pd_vts
        for i in range(ctx['spring_opt_iters']):
            x=proj_module(x)
            cr_vts_list.append(x)
        gt_vts=gt_vt_offsets+skin
        avg_loss=loss_manager.get_avg_mse_loss(cr_vts_list,gt_vts,norm_vts=False)
        loss+=avg_loss
        loss_manager.add_item_loss('avg_loss',avg_loss.item()*n_samples) # to equally compare with previous results

        pre_proj_loss=loss_manager.get_mse_loss(pd_vt_offsets,gt_vt_offsets,norm_vts=False)
        loss_manager.add_item_loss('pre_proj_loss',pre_proj_loss.item()*n_samples)
        post_proj_loss=loss_manager.get_mse_loss(cr_vts_list[-1],gt_vts,norm_vts=False)
        loss_manager.add_item_loss('post_proj_loss',post_proj_loss.item()*n_samples)

    elif ctx['use_vt_loss']:
        if not ctx['pd_only']:
            vt_loss=loss_manager.get_mse_loss(pd_vt_offsets,gt_vt_offsets,norm_vts=False)
            loss+=vt_loss
            loss_manager.add_item_loss('vt_loss',vt_loss.item()*n_samples)
    else:
        if not ctx['pd_only']:
            if ctx['use_patches']:
                pix_loss=loss_manager.get_pix_loss(pd_offset_imgs,gt_offset_imgs,crop_masks,normalize=False)
                pix_loss/=torch.sum(crop_masks)/len(res_ctx['vt_ids_in_crop'])
            else:
                pix_loss=loss_manager.get_pix_loss_both_sides(pd_offset_imgs,gt_offset_imgs,front_masks,back_masks)
                # pix_loss/=(torch.sum(front_masks[0,0,:,:])+torch.sum(back_masks[0,0,:,:]))*n_samples

            loss+=pix_loss
            loss_manager.add_item_loss('pix_loss',pix_loss.item()*n_samples)

        if ctx['calc_vt_loss']:
            vt_loss=loss_manager.get_vt_loss(pd_vt_offsets,gt_vt_offsets)
            loss+=vt_loss
            loss_manager.add_item_loss('vt_loss',vt_loss.item()*n_samples)

        if ctx['use_mix'] and mode=='train' and batch['label']=='midres':
            loss=loss*ctx['lambda_mix']

    if not ctx['pd_only']:
        loss_manager.add_total_loss(loss.item()*n_samples)
        loss_manager.add_samples(n_samples)

    # step 
    if mode=='train':
        if ctx['use_variable_m']:
            m_opt.zero_grad()
            opt.zero_grad()
            loss.backward()
            m_opt.step()
            opt.step()
        else:
            opt.zero_grad()
            loss.backward()
            opt.step()


    # tensorboard visualization
    if vlz:
        gt_offset_img=gt_offset_imgs[0].detach().cpu().numpy()
        pd_offset_img=pd_offset_imgs[0].detach().cpu().numpy()
        if not ctx['use_patches']:
            front_mask=front_masks[0].squeeze().cpu().numpy()
            back_mask=back_masks[0].squeeze().cpu().numpy()
            arr_stats={'minval':np.full(6,np.min(gt_offset_img)),'maxval':np.full(6,np.max(gt_offset_img))}
            # print(gt_offset_img[0,140:146,140:146])
            # print(np.mean(gt_offset_img[0,:,:]))
            # print(np.amax(gt_offset_img[0,:,:]))
            # print(np.amin(gt_offset_img[0,:,:]))
            # print(pd_offset_img[0,:,:])
            # print(np.mean(pd_offset_img[0,:,:]))
            # print(np.amax(pd_offset_img[0,:,:]))
            # print(np.amin(pd_offset_img[0,:,:]))
            vlz_img=vlz_pd_offset_img_both_sides(pd_offset_img,gt_offset_img,front_mask,back_mask,arr_stats=arr_stats)
        else:
            crop_mask=crop_masks[0].squeeze().cpu().numpy()
            arr_stats={'minval':np.full(3,np.min(gt_offset_img)),'maxval':np.full(3,np.max(gt_offset_img))}
            vlz_img=vlz_pd_offset_img(pd_offset_img,gt_offset_img,crop_mask,arr_stats=arr_stats)
        ctx['tb_logger'].image_summary('{}_img'.format(mode),np.expand_dims(vlz_img,axis=0),epoch)

    # write output
    if ctx['use_gen_diff']:
        write_diff_imgs(batch,gt_offset_imgs-pd_offset_imgs,front_masks,back_masks,res_ctx['diff_img_dir'])
    if mode=='eval' and ctx['write_pd']:
        assert(not ctx['use_gen_diff'])
        if ctx['use_patches']:
            if ctx['write_pd_skin_only']:
                write_offsets(batch,pd_vts.detach().cpu(),out_dir=ctx['eval_out_dir'],prefix='skin')
                write_offsets(batch,batch['vt_offset']-pd_vt_offsets.detach().cpu(),out_dir=ctx['eval_out_dir'],prefix='offset')
            else:
                write_objs(batch,batch['vt_offset']+batch['skin'],fc=res_ctx['patch_local_fcs'],out_dir=ctx['eval_out_dir'],prefix='gt')
                write_objs(batch,pd_vt_offsets.detach().cpu()+batch['skin'],fc=res_ctx['patch_local_fcs'],out_dir=ctx['eval_out_dir'],prefix='pd')
                if ctx['use_cvxpy'] or ctx['use_ecos']:
                    write_objs(batch,cr_vts,fc=res_ctx['patch_local_fcs'],out_dir=ctx['eval_out_dir'],prefix='cr')
        else:
            # write_output(batch,pd_vt_offsets,out_dir=ctx['eval_out_dir'],offset_io_manager=managers['offset_io_manager'])
            if not ctx['pd_only']:
                write_cloth(batch,batch['vt_offset'],out_dir=ctx['eval_out_dir'],offset_io_manager=managers['offset_io_manager'],prefix='gt_cloth')
            write_cloth(batch,pd_vt_offsets,out_dir=ctx['eval_out_dir'],offset_io_manager=managers['offset_io_manager'],prefix='pd_cloth')
            if ctx['use_cvxpy'] or ctx['use_ecos'] or ctx['use_cudaqs']:
                write_cloth(batch,cr_vt_offsets,out_dir=ctx['eval_out_dir'],offset_io_manager=managers['offset_io_manager'],prefix='cr_cloth')
                # gt_vts=gt_vt_offsets+skin
                # cr_gt_vts=proj_module(gt_vts)
                # cr_gt_vt_offsets=cr_gt_vts-skin
                # write_cloth(batch,cr_gt_vt_offsets,out_dir=ctx['eval_out_dir'],offset_io_manager=managers['offset_io_manager'],prefix='cr_gt_cloth')
                # write_npys(batch,pd_vts,out_dir=ctx['eval_out_dir'],prefix='pd_vts')
                if ctx['use_variable_m']:
                    write_npys(batch,pd_ms,out_dir=ctx['eval_out_dir'],prefix='pd_m')
        if ctx['write_pd_imgs']:
            write_imgs(batch,pd_offset_imgs,out_dir=ctx['eval_out_dir'])

    if ctx['use_debug']:
        print('ids:',batch['index'].tolist())
        assert(False)

    if (ctx['use_cvxpy'] or ctx['use_ecos'] or ctx['use_cudaqs']) and not ctx['pd_only']:
        return post_proj_loss
    else:
        return loss

def process_epoch(dataloader,net,opt,loss_fn,epoch,ctx,mode='train'):
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
            'offset_manager':OffsetManager(shared_data_dir=res_ctx['shared_data_dir'],ctx=ctx),
            'loss_manager':LossManager(ctx=ctx)
        }
        if mode=='eval':
            managers['offset_io_manager']=OffsetIOManager(res_ctx=res_ctx,ctx=ctx)
        if ctx['use_cvxpy'] or ctx['calc_vt_loss']:
            managers['img_sample_module']=ImgSampleBothSidesModule(managers['offset_manager'])
            # managers['img_sample_module']=ImgSample2ndOrderBothSidesModule(managers['offset_manager'])
        return managers

    if ctx['use_mix']:
        if mode=='train':
            res_managers={res_name:get_managers(res_ctx) for res_name,res_ctx in ctx['mixres_ctxs'].items()}
            mixres_ctxs=ctx['mixres_ctxs']
            dataloader.reset()
        else:
            managers=get_managers(ctx['mixres_ctxs']['midres'])
            res_ctx=ctx['mixres_ctxs']['midres']
    else:
        managers=get_managers(ctx['res_ctx'])
        res_ctx=ctx['res_ctx']

    iter_start_time=time.time()
    # process batches
    for idx,batch in enumerate(dataloader):
        if ctx['use_mix'] and mode=='train':
            managers=res_managers[batch['label']]
            res_ctx=mixres_ctxs[batch['label']]

        vlz=epoch%ctx['vlz_per_epoch']==0 and idx==0 and (mode=='train' or mode=='validate')
        process_batch(batch,net,opt,managers,epoch,mode=mode,vlz=vlz,res_ctx=res_ctx,ctx=ctx)

        if ctx['print_per_iter']>0 and idx%ctx['print_per_iter']==0:
            item_loss=managers['loss_manager'].get_item_loss()
            iter_end_time=time.time()
            # print('iter_time',iter_end_time-iter_start_time,'s')
            # managers['loss_manager'].print_item_loss(item_loss)
            iter_start_time=time.time()

        if ctx['max_iter_per_epoch']>=0 and idx>=ctx['max_iter_per_epoch']:
            break

    # log loss
    if mode=='train' or mode=='validate':
        def log_loss(loss_manager,tb_logger,surfix=''):
            item_loss=loss_manager.get_item_loss()
            for name,loss in item_loss.items():
                tb_logger.scalar_summary('{}_{}{}'.format(mode,name,surfix),loss,epoch)
            total_loss=loss_manager.get_total_loss()
            tb_logger.scalar_summary('{}_total{}'.format(mode,surfix),total_loss,epoch)

        if ctx['use_mix'] and mode=='train':
            for res_name,managers in res_managers.items():
                log_loss(managers['loss_manager'],ctx['tb_logger'],surfix='_{}'.format(res_name))
        else:
            log_loss(managers['loss_manager'],ctx['tb_logger'])

    # print necessary information
    epoch_end_time=time.time()
    if ctx['pd_only']:
        return 0
    
    if ctx['use_mix'] and mode=='train':
        total_loss=0
        total_samples=0
        # average all losses
        for name,managers in res_managers.items():
            loss_manager=managers['loss_manager']
            total_loss+=loss_manager.get_total_loss()
            total_samples+=loss_manager.total_samples
        total_loss/=total_samples
    else:
        total_loss=managers['loss_manager'].get_total_loss()

    if epoch%ctx['print_per_epoch']==0:
        print(mode,'epoch:',epoch,', time:',time.strftime('%H:%M:%S',time.gmtime(epoch_end_time-epoch_start_time)),'total_loss',total_loss)
        item_loss=managers['loss_manager'].get_item_loss()
        managers['loss_manager'].print_item_loss(item_loss)
        if ctx['write_file_log']:
            ctx['file_logger'].log('{}_total'.format(mode),total_loss,epoch)

    managers['loss_manager'].clear()

    return total_loss

def save_model(model, optimizer, lr, val_cost, epoch, save_path, ctx):
    state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lr': lr,
            'val_cost': val_cost,
            'optimizer' : optimizer.state_dict(),
            }
    if ctx['use_variable_m']:
        state['m_net']=ctx['m_net'].state_dict()
        state['m_opt']=ctx['m_opt'].state_dict()
    torch.save(state, save_path)

# def load_model(model, optimizer, lr, save_path):
#     state = torch.load(save_path)
#     model.load_state_dict(state['state_dict'])
#     if lr == state['lr']:
#         optimizer.load_state_dict(state['optimizer'])
#     else:
#         print('different learning rate %f from before %f, not loading optimzer state...' %(lr, state['lr']))


def train_model(dataloaders,net,opt,loss_fn,start_epoch,ctx):
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
    validate_loss=0

    for epoch in range(start_epoch,num_epochs):
        train_loss=process_epoch(dataloaders['train'],net,opt,loss_fn,epoch,ctx,mode='train')
        if not ctx['skip_val']:
            validate_loss=process_epoch(dataloaders['val'],net,opt,loss_fn,epoch,ctx,mode='validate')

            if validate_loss<best_val_loss and epoch>ctx['save_after_epoch']:
                print('validate loss is lower than best before, saving model...')
                best_val_loss=validate_loss
                save_path=join(save_dir,'val_model_best.pth.tar')
                save_model(net,opt,ctx['lr'],validate_loss,epoch,save_path,ctx)
                print('save model:',save_path)

        if train_loss<best_train_loss and epoch>ctx['save_after_epoch']:
            print('train loss is lower than best before, saving model...')
            best_train_loss=train_loss
            save_path=join(save_dir,'train_model_best.pth.tar')
            save_model(net,opt,ctx['lr'],train_loss,epoch,save_path,ctx)

        if (epoch+1)%save_every_epoch==0 and epoch>ctx['save_after_epoch']:
            save_path=join(save_dir,'checkpoint-{}.pth.tar'.format(epoch+1))
            save_model(net,opt,ctx['lr'],validate_loss,epoch,save_path,ctx)

def eval_model(dataloader,net,loss_fn,ctx):
    process_epoch(dataloader,net,opt=None,loss_fn=loss_fn,epoch=0,ctx=ctx,mode='eval')

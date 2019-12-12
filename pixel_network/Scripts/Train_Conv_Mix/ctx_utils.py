######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,isdir,exists
from patch_utils import PatchManager
import torch
import torch.nn.functional as F
import numpy as np
from cvxpy_opt import CvxpyOpt
from cvxpy_opt_func import init_cvxpy_opt_module
from ecos_opt_func import init_ecos_opt_module
from spring_opt_func import SpringOptModule
from cudaqs_func import init_cudaqs_module
from inequality_opt import InequalityOptSystem
from obj_io import Obj,read_obj,write_obj
import functools
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-test_case',choices=['lowres','midres','mixres','midres_pgan','hero_pgan','midres_acgan_regress','midres_odenet','midres_style_gan','hero_style_gan', 'lowres_style_gan','midres_gen_diff','midres_diff','lowres_cvxpy','lowres_ecos','lowres_vt','lowres_patch','lowres_ecos_patch','lowres_vt_patch','midres_patch','midres_ecos_patch','midres_vt_patch','lowres_patch','lowres_spring_patch','lowres_cudaqs', 'lowres_tex', 'lowres_tex_vt', 'highres_tex'])
parser.add_argument('-trial',default='test')
parser.add_argument('-device',default='cuda:0')
parser.add_argument('-data_root_dir',default='/data/poses_v3')
parser.add_argument('-run_root_dir',default=None)
parser.add_argument('-init_linear_layers',type=int,default=0) # previous: 1
parser.add_argument('-use_coord_conv',action='store_true')
parser.add_argument('-use_up_conv',action='store_true')
parser.add_argument('-use_skip_link',action='store_true')
parser.add_argument('-use_dropout',action='store_true')
parser.add_argument('-n_res_blocks',type=int,default=0)
parser.add_argument('-init_channels',type=int,default=256) # previous: 64
parser.add_argument('-init_size',type=int,default=8) # previous: 16
parser.add_argument('-max_num_samples',type=int,default=-1)
parser.add_argument('-show_every_epoch',type=int,default=10)
parser.add_argument('-lr',type=float,default=1e-3)
parser.add_argument('-eval',choices=['train','test','none','val'],default='none')
parser.add_argument('-eval_disable_output',action='store_true')
parser.add_argument('-eval_write_offsets',action='store_true')
parser.add_argument('-cp',default='')
parser.add_argument('-batch_size',type=int,default=64)
parser.add_argument('-num_workers',type=int,default=8)
parser.add_argument('-weight_decay',type=float,default=0)
parser.add_argument('-use_uvn',action='store_true')
parser.add_argument('-uvn_dir',default=None)
parser.add_argument('-offset_img_dir',default=None)
parser.add_argument('-vt_offset_dir',default=None)
parser.add_argument('-skin_dir',default=None)
parser.add_argument('-vn_dir',default=None)
parser.add_argument('-use_normals',action='store_true')
parser.add_argument('-no_shuffle',action='store_true')
parser.add_argument('-rotation_matrices_dir',default='rotation_matrices')
parser.add_argument('-loss_type',choices=['l1','l2','mse'],default='mse')
parser.add_argument('-offset_img_size',type=int,default=512)
parser.add_argument('-sample_list_file',default=None)
parser.add_argument('-save_every_epoch',type=int,default=50)
parser.add_argument('-lambda_mix',type=float,default=4)
parser.add_argument('-use_multi_layer_loss',action='store_true')
parser.add_argument('-dtype',choices=['float','double'],default='double')
parser.add_argument('-use_trace',action='store_true')
parser.add_argument('-num_epochs',type=int,default=1000000)
parser.add_argument('-sample_list_prefix',default=None)
parser.add_argument('-pd_only',action='store_true')
parser.add_argument('-dir_surfix',default=None)
parser.add_argument('-start_epoch',type=int,default=0)
parser.add_argument('-print_per_epoch',type=int,default=1)
parser.add_argument('-print_per_iter',type=int,default=-1)
parser.add_argument('-vlz_per_epoch',type=int,default=1)
parser.add_argument('-max_iter_per_epoch',type=int,default=-1)
parser.add_argument('-relu',choices=['relu','prelu'],default='relu')
parser.add_argument('-write_pd_imgs',action='store_true')

# patch utils
parser.add_argument('-use_patches',action='store_true')
parser.add_argument('-patch_id',type=int,default=-1)

# progressive gan
parser.add_argument('-start_level',type=int,default=3)
parser.add_argument('-end_level',type=int,default=8)
parser.add_argument('-epochs_per_level',type=int,default=20)
parser.add_argument('-n_critic',type=int,default=1)
parser.add_argument('-channelsD',default='v3')
parser.add_argument('-channelsG',default='v2')
parser.add_argument('-lambda_gp',type=float,default=0)
parser.add_argument('-lambda_l1',type=float,default=0)
parser.add_argument('-lambda_R1',type=float,default=10)
parser.add_argument('-lambda_R2',type=float,default=0)
parser.add_argument('-lambda_sim_l1',type=float,default=0)
parser.add_argument('-lambda_consensus',type=float,default=0)
parser.add_argument('-lambda_proj',type=float,default=1)
parser.add_argument('-lambda_zgp',type=float,default=0)

# normalize input/output
parser.add_argument('-normalize_rotations',action='store_true')
parser.add_argument('-normalize_offset_imgs',action='store_true')

# use catogorical D
parser.add_argument('-use_ctgr_D',action='store_true')
parser.add_argument('-lambda_ctgr_sim',type=float,default=1)
parser.add_argument('-lambda_Rx_sim',type=float,default=1)
# skin imgs
parser.add_argument('-cat_skin_imgs',action='store_true')
parser.add_argument('-normalize_skin_imgs',action='store_true')
# use catogorical D
parser.add_argument('-use_ctgr_D_v2',action='store_true')
parser.add_argument('-use_ctgr_D_v3',action='store_true')
parser.add_argument('-use_lsgan',action='store_true')
parser.add_argument('-use_style_gan',action='store_true')
parser.add_argument('-use_proj',action='store_true') # style gan specific
# pairwise D
parser.add_argument('-use_pair_D',action='store_true')
parser.add_argument('-save_after_epoch',type=int,default=-1)
parser.add_argument('-write_file_log',action='store_true')
parser.add_argument('-use_label',action='store_true')
parser.add_argument('-use_gz',action='store_true')
# cvx
parser.add_argument('-max_num_constraints',type=int,default=-1)
parser.add_argument('-skip_val',action='store_true')
parser.add_argument('-use_lap',action='store_true')
parser.add_argument('-lmd_lap',type=float,default=1)
parser.add_argument('-use_avg_loss',action='store_true')
parser.add_argument('-write_pd_skin_only',action='store_true')
parser.add_argument('-eval_out_dir',default=None)
parser.add_argument('-use_spring',action='store_true')
parser.add_argument('-lmd_k',type=float,default=5e-2)
parser.add_argument('-use_debug',action='store_true')
parser.add_argument('-use_variable_m',action='store_true')
# spring
parser.add_argument('-spring_opt_iters',type=int,default=10)
parser.add_argument('-stiffen_anchor_factor',type=float,default=0.2)

args=parser.parse_args()

test_case=args.test_case
trial=args.trial

# directories
project_root_dir='../..'
learning_root_dir=join(project_root_dir,'Learning')
data_root_dir=args.data_root_dir
assert(isdir(data_root_dir))

if args.run_root_dir is None:
    run_test_case_dir=os.path.join(learning_root_dir,'rundir',args.test_case)
else:
    run_test_case_dir=os.path.join(args.run_root_dir,args.test_case)
# if args.use_patches:
#     run_test_case_dir+='_patches'
if not exists(run_test_case_dir):
    os.makedirs(run_test_case_dir)
rundir=join(run_test_case_dir,trial)
if not exists(rundir):
    os.makedirs(rundir)

# misc
device=torch.device(args.device)
print('device',device)

max_num_samples=args.max_num_samples
batch_size=args.batch_size
num_workers=args.num_workers
offset_img_size=args.offset_img_size

input_size=90
learning_rate=args.lr
dtype=torch.float if args.dtype=='float' else torch.double

use_uvn=args.use_uvn
# use_uvn=True
use_normals=False
if args.use_normals:
    use_normals=True
if test_case.find('normal')!=-1:
    use_normals=True
use_mix=test_case.find('mixres')!=-1
use_patches=args.use_patches or test_case.find('patch')!=-1
use_pgan=test_case.find('pgan')!=-1 or test_case.find('style_gan')!=-1
if use_pgan:
    offset_img_size=2**args.end_level
    dtype=torch.float
    # num_epochs=(args.end_level-args.start_level+1)*args.epochs_per_level
use_hero=test_case.find('hero')!=-1
use_style_gan=test_case.find('style_gan')!=-1
use_conv=test_case=='midres' or test_case=='lowres' or test_case=='midres_diff' or test_case=='midres_gen_diff' or test_case=='lowres_cvxpy' or test_case=='lowres_ecos' or test_case=='lowres_vt' or test_case=='lowres_patch' or test_case=='lowres_ecos_patch' or test_case=='lowres_vt_patch' or test_case=='midres_patch' or test_case=='midres_ecos_patch' or test_case=='midres_vt_patch' or test_case=='lowres_spring' or test_case=='lowres_spring_patch' or test_case=='lowres_cudaqs' or test_case=='lowres_tex' or test_case=='lowres_tex_vt' or test_case=='highres_tex'

use_cvxpy=test_case.find('cvxpy')!=-1
use_ecos=test_case.find('ecos')!=-1
use_spring=test_case.find('spring')!=-1
use_cudaqs=test_case.find('cudaqs')!=-1

if use_mix:
    sample_list_prefix='mix_'
elif use_hero:
    sample_list_prefix='hero_'
else:
    sample_list_prefix=''
if args.sample_list_prefix is not None:
    sample_list_prefix=args.sample_list_prefix
    if not sample_list_prefix.endswith('_'):
        sample_list_prefix=sample_list_prefix+'_'

pd_only=args.pd_only
if args.dir_surfix is not None:
    dir_surfix=args.dir_surfix
    if not dir_surfix.startswith('_'):
        dir_surfix='_{}'.format(dir_surfix)
else:
    dir_surfix=''

use_acgan=test_case.find('acgan')!=-1
use_regress=test_case.find('regress')!=-1
if use_acgan:
    dtype=torch.float

use_odenet=test_case.find('odenet')!=-1

use_gen_diff=test_case.find('gen_diff')!=-1
use_diff=test_case.find('diff')!=-1 and test_case.find('gen_diff')==-1

rotation_mat_dir=join(data_root_dir,'rotation_matrices{}'.format(dir_surfix))

# test case specific arguments
def get_res_ctx(res):
    if use_uvn:
        offset_img_dir='{}_uvn_offset_imgs_{}{}'.format(res,offset_img_size,dir_surfix) 
    elif args.test_case=='lowres_tex' or args.test_case=='lowres_tex_vt' or args.test_case=='highres_tex':
        offset_img_dir='{}_texture_imgs_{}{}'.format(res,offset_img_size,dir_surfix) 
    else:
        offset_img_dir='{}_offset_imgs_{}{}'.format(res,offset_img_size,dir_surfix)
    skin_img_dir='{}_skin_imgs_{}{}'.format(res,offset_img_size,dir_surfix)
    shared_data_dir=join(learning_root_dir,'shared_data_{}'.format(res))

    if args.sample_list_file is None:
        sample_list_file_dict={name:join(data_root_dir,'sample_lists/{}{}_{}_samples.txt'.format(sample_list_prefix,res,name)) for name in {'train','val','test'}}
    else:
        sample_list_file_dict={name:join(data_root_dir,'sample_lists/{}'.format(args.sample_list_file)) for name in {'train','val','test'}}

    res_ctx={
        'shared_data_dir':shared_data_dir,
        'sample_list_file_dict':sample_list_file_dict,
        'uvn_dir':join(data_root_dir,'{}_skin_tshirt_nuvs{}'.format(res,dir_surfix)),
        'offset_img_dir':join(data_root_dir,offset_img_dir),
        'vt_offset_dir':join(data_root_dir,'{}_offset_npys{}'.format(res,dir_surfix)),
        'skin_dir':join(data_root_dir,'{}_skin_npys{}'.format(res,dir_surfix)),
        'vn_dir':join(data_root_dir,'{}_normal_npys{}'.format(res,dir_surfix)),
        'mask_file':join(learning_root_dir,'shared_data_{}'.format(res),'offset_img_mask_{}.npy'.format(offset_img_size)),
        'skin_img_dir':join(data_root_dir,skin_img_dir)
    }

    if args.test_case == 'lowres_tex' or args.test_case == 'lowres_tex_vt' or args.test_case == 'highres_tex':
        res_ctx['vt_offset_dir']=join(data_root_dir,'{}_texture_txt{}'.format(res,dir_surfix))

    if use_gen_diff or use_diff:
        diff_img_dir='{}_uvn_diff_imgs_{}{}'.format(res,offset_img_size,dir_surfix) if use_uvn else '{}_diff_imgs_{}{}'.format(res,offset_img_size,dir_surfix)
        res_ctx['diff_img_dir']=join(data_root_dir,diff_img_dir)

    if use_patches:
        original_size=512
        patch_manager=PatchManager(shared_data_dir=shared_data_dir)
        res_ctx['patch_manager']=patch_manager
        patch_id=args.patch_id
        crop=patch_manager.load_patch_crop(patch_manager.get_patch_crop_path(patch_id))
        vt_ids_in_crop=patch_manager.load_patch_vt_ids(patch_id)
        vts_in_crop=patch_manager.get_vts_in_crop(vt_ids_in_crop,crop,original_size=(original_size,original_size))
        vts_in_crop=torch.from_numpy(vts_in_crop).to(device)
        res_ctx['vt_ids_in_crop']=vt_ids_in_crop
        res_ctx['vts_in_crop']=vts_in_crop
        global_fcs=patch_manager.fcs
        patch_fc_ids=patch_manager.get_patch_fc_ids(vt_ids_in_crop,global_fcs)
        patch_global_fcs=global_fcs[patch_fc_ids]
        res_ctx['patch_local_fcs']=patch_manager.get_patch_local_fcs(vt_ids_in_crop,patch_global_fcs)
        # res_ctx['patch_local_fcs']=patch_manager

    return res_ctx

if use_hero:
    res_ctx=get_res_ctx('midres')
    # res_ctx=get_res_ctx('lowres')
    sim_res_ctx=get_res_ctx('lowres')
elif test_case.find('lowres')!=-1:
    res_ctx=get_res_ctx('lowres')
elif test_case.find('midres')!=-1:
    res_ctx=get_res_ctx('midres')
elif test_case.find('highres')!=-1:
    res_ctx=get_res_ctx('highres')
elif test_case.find('mixres')!=-1:
    mixres_ctxs={res:get_res_ctx(res) for res in ['midres','lowres']}


def overwrite_res_ctx(res_ctx):
    if args.uvn_dir is not None:
        res_ctx['uvn_dir']=args.uvn_dir
    if args.offset_img_dir is not None:
        res_ctx['offset_img_dir']=args.offset_img_dir
    if args.vt_offset_dir is not None:
        res_ctx['vt_offset_dir']=args.vt_offset_dir
    if args.skin_dir is not None:
        res_ctx['skin_dir']=args.skin_dir
    if args.vn_dir is not None:
        res_ctx['vn_dir']=args.vn_dir
    return res_ctx


def check_dir(d,fatal=True):
    if not isdir(d):
        print(d,'is not a directory')
        if fatal:
            assert(False)

def check_res_ctx(res_ctx):
    check_dir(res_ctx['offset_img_dir'],fatal=not pd_only)
    check_dir(res_ctx['vt_offset_dir'],fatal=not pd_only)
    check_dir(res_ctx['skin_dir'],fatal=use_normals)
    check_dir(res_ctx['vn_dir'],fatal=use_normals)
    check_dir(res_ctx['uvn_dir'],fatal=use_uvn)

if not use_mix:
    overwrite_res_ctx(res_ctx)
    check_res_ctx(res_ctx)
else:
    for res_name,res_ctx in mixres_ctxs.items():
        check_res_ctx(res_ctx)

# patches
if use_patches:
    patch_id=args.patch_id
    original_size=512
    init_size=5
    if use_mix:
        patch_manager=mixres_ctxs['lowres']['patch_manager']
    else:
        patch_manager=res_ctx['patch_manager']
    crop_size=patch_manager.crop_size
    n_patches=len(patch_manager.patch_names)
    assert(patch_id>=0 and patch_id<n_patches)
    patch_name=patch_manager.patch_names[patch_id]
    crop=patch_manager.load_patch_crop(patch_manager.get_patch_crop_path(patch_id))
    crop_mask=np.load(patch_manager.get_patch_mask_path(patch_id))
    print('load patch',patch_name,'crop',crop)
    rundir=join(run_test_case_dir,args.trial,patch_name)

args_eval_out_dir=args.eval_out_dir
vt_offset_dir_is_set=args.vt_offset_dir is not None

# export information
ctx=vars(args)
ctx['project_root_dir']=project_root_dir
ctx['learning_root_dir']=learning_root_dir
ctx['data_root_dir']=data_root_dir
ctx['rotation_mat_dir']=rotation_mat_dir
ctx['rundir']=rundir
ctx['device']=device
ctx['input_size']=input_size
ctx['use_uvn']=use_uvn
ctx['use_mix']=use_mix
ctx['use_normals']=use_normals
ctx['print_every_epoch']=-1
ctx['eval_out_dir']=join(rundir,'eval_{}'.format(args.eval))
ctx['write_pd']=True
ctx['dtype']=dtype

ctx.pop('shared_data_dir',None)
ctx.pop('uvn_dir',None)
ctx.pop('offset_img_dir',None)
ctx.pop('vt_offset_dir',None)
ctx.pop('skin_dir',None)
ctx.pop('vn_dir',None)

if not use_mix:
    ctx['res_ctx']=res_ctx
else:
    ctx['mixres_ctxs']=mixres_ctxs

ctx['use_hero']=use_hero
if use_hero:
    ctx['sim_res_ctx']=sim_res_ctx

ctx['use_patches']=use_patches
if use_patches:
    ctx['patch_id']=patch_id
    ctx['n_patches']=n_patches
    ctx['patch_name']=patch_name
    ctx['crop']=crop
    ctx['crop_mask']=crop_mask
    ctx['original_size']=original_size
    ctx['crop_size']=crop_size
    ctx['init_size']=init_size
    ctx['eval_out_dir']=join(run_test_case_dir,args.trial,'eval_{}'.format(args.eval),patch_name)
    print('init_size',init_size,'crop_size',crop_size)

if args_eval_out_dir is not None:
    ctx['eval_out_dir']=args_eval_out_dir

def add_level_masks(res_ctx):
    mask=torch.from_numpy(np.load(res_ctx['mask_file'])).squeeze().to(device=device,dtype=dtype)
    front_mask=mask[:,:,0].view(1,1,mask.size(0),mask.size(1))
    back_mask=mask[:,:,1].view(1,1,mask.size(0),mask.size(1))
    res_ctx['level_to_front_mask']=[None for i in range(end_level+1)]
    res_ctx['level_to_back_mask']=[None for i in range(end_level+1)]
    for level in reversed(range(start_level,end_level+1)):
        res_ctx['level_to_front_mask'][level]=F.interpolate(front_mask,scale_factor=2**(level-end_level))
        res_ctx['level_to_back_mask'][level]=F.interpolate(back_mask,scale_factor=2**(level-end_level))

ctx['use_conv']=use_conv
if use_conv:
    ctx['calc_vt_loss']=False
    # ctx['calc_vt_loss']=True

ctx['use_pgan']=use_pgan
if use_pgan:
    start_level=ctx['start_level']
    end_level=ctx['end_level']
    ctx['channelsG']=np.loadtxt('cfgs/channelsG_{}.txt'.format(args.channelsG)).astype(np.uint32)
    ctx['channelsD']=np.loadtxt('cfgs/channelsD_{}.txt'.format(args.channelsD)).astype(np.uint32)
    ctx['normalize_rotations']=True
    ctx['normalize_offset_imgs']=True
    ctx['normalize_skin_imgs']=True
    if not use_patches:
        add_level_masks(res_ctx)
        if use_hero:
            add_level_masks(sim_res_ctx)
    else:
        assert(False)

ctx['use_style_gan']=use_style_gan
# ctx['use_label']=False
ctx['cat_label']=False
ctx['use_separate_block']=False
if use_style_gan:
    ctx['use_coord_conv']=True
    if use_hero:
        # ctx['cat_label']=True
        ctx['use_separate_block']=True
        if ctx['cat_label']:
            ctx['input_size']+=2

ctx['use_acgan']=use_acgan
ctx['use_regress']=use_regress
if use_acgan and use_regress:
    ctx['use_coord_conv']=True
    ctx['normalize_rotations']=True
    ctx['normalize_offset_imgs']=True

ctx['use_odenet']=use_odenet
if use_odenet:
    ctx['loss_type']='mse'
    ctx['normalize_rotations']=True
    ctx['normalize_offset_imgs']=True

ctx['use_gen_diff']=use_gen_diff
if use_gen_diff:
    if not isdir(res_ctx['diff_img_dir']):
        os.makedirs(res_ctx['diff_img_dir'])
    ctx['calc_vt_loss']=False
    ctx['write_pd']=False
ctx['use_diff']=use_diff

if ctx['use_debug']:
    debug_dir='debug'
    ctx['debug_dir']=debug_dir
    if not isdir(debug_dir):
        print('create:',debug_dir)
        os.makedirs(debug_dir)
    else:
        bak_i=1
        while True:
            debug_bak_dir='{}_bak_{}'.format(debug_dir,bak_i)
            if not isdir(debug_bak_dir):
                cmd='mv {} {}'.format(debug_dir,debug_bak_dir)
                print(cmd)
                os.system(cmd)
                print('create:',debug_dir)
                os.makedirs(debug_dir)
                break    
            bak_i+=1
    ctx['verbose']=True

if args.test_case=='lowres_tex' or args.test_case == 'lowres_tex_vt' or args.test_case == 'highres_tex':
    ctx['output_channels']=4 if not ctx['use_patches'] else 2
else:
    ctx['output_channels']=6 if not ctx['use_patches'] else 3

ctx['use_cvxpy']=use_cvxpy
if use_cvxpy:
    ctx['cvxpy_module']=init_cvxpy_opt_module(res_ctx,ctx)
ctx['use_ecos']=use_ecos
if use_ecos:
    ctx['ecos_module']=init_ecos_opt_module(res_ctx,ctx)
ctx['use_spring']=use_spring
if use_spring:
    ctx['spring_module']=SpringOptModule(res_ctx,ctx)
ctx['use_cudaqs']=use_cudaqs
if use_cudaqs:
    ctx['cudaqs_module']=init_cudaqs_module(res_ctx,ctx)

if ctx['use_variable_m']:
    ctx['m_init_channels']=ctx['offset_img_size']//ctx['init_size']
    ctx['m_output_channels']=2 if not ctx['use_patches'] else 1

ctx['load_opt']=True
if use_cvxpy or use_ecos or use_cudaqs:
    pass
    # ctx['load_opt']=False
if ctx['use_ecos']:
    # pass
    if test_case.find('lowres')!=-1 and not vt_offset_dir_is_set:
        res_ctx['vt_offset_dir']=join(data_root_dir,'lowres_offsets_len-2lap1')
    elif test_case.find('midres')!=-1 and not vt_offset_dir_is_set:
        res_ctx['vt_offset_dir']=join(data_root_dir,'midres_offsets_len-2lap-1')
if ctx['use_spring']:
    if test_case.find('lowres')!=-1 and not vt_offset_dir_is_set:
        res_ctx['vt_offset_dir']=join(data_root_dir,'lowres_offsets_i10')
ctx['use_vt_loss']=False
if test_case=='lowres_vt' or test_case=='lowres_vt_patch' or test_case=='midres_vt' or test_case=='midres_vt_patch' or args.test_case == 'lowres_tex_vt':
    ctx['use_vt_loss']=True
    ctx['load_opt']=False
    # for fairness
    # print('args.vt_offset_dir',args.vt_offset_dir)
    # assert(False)
    # if test_case.find('lowres')!=-1 and not vt_offset_dir_is_set:
    #     res_ctx['vt_offset_dir']=join(data_root_dir,'lowres_offsets_len-2lap1') 
    # elif test_case.find('midres')!=-1 and not vt_offset_dir_is_set:
    #     res_ctx['vt_offset_dir']=join(data_root_dir,'midres_offsets_len-2lap-1')

ctx['load_vt_offset']=(ctx['eval']!='none') or ctx['use_cvxpy'] or ctx['use_ecos'] or ctx['use_spring'] or ctx['use_cudaqs'] or ctx['calc_vt_loss'] or ctx['use_vt_loss']
ctx['load_skin']=ctx['use_normals'] or ctx['use_cvxpy'] or ctx['use_ecos'] or ctx['use_spring'] or ctx['use_cudaqs'] or (ctx['eval']!='none')

def add_mask(res_ctx):
    mask=torch.from_numpy(np.load(res_ctx['mask_file'])).squeeze().to(device=device,dtype=dtype)
    res_ctx['front_mask']=mask[:,:,0].view(1,1,mask.size(0),mask.size(1))
    res_ctx['back_mask']=mask[:,:,1].view(1,1,mask.size(0),mask.size(1))

add_mask(res_ctx)
if ctx['use_hero']:
    add_mask(sim_res_ctx)

# normalize mean & std
def add_img_stats(res_ctx,img_type='offset_img'):
    shared_data_dir=res_ctx['shared_data_dir']
    mean=torch.from_numpy(np.load(join(shared_data_dir,'{}_{}_mean.npy'.format(img_type,offset_img_size)))).permute(2,0,1)

    print('mean',mean.size())
    res_ctx['{}_mean'.format(img_type)]=mean.view(1,mean.size(0),mean.size(1),mean.size(2)).to(device=device,dtype=dtype)
    std=torch.from_numpy(np.load(join(shared_data_dir,'{}_{}_std.npy'.format(img_type,offset_img_size)))).permute(2,0,1)
    std[std<1e-6]=1 # avoid divide by 0
    res_ctx['{}_std'.format(img_type)]=std.view(1,std.size(0),std.size(1),std.size(2)).to(device=device,dtype=dtype)

if ctx['normalize_offset_imgs']:
    add_img_stats(res_ctx)
    if ctx['use_hero']:
        add_img_stats(sim_res_ctx)

if ctx['normalize_skin_imgs']:
    add_img_stats(res_ctx,img_type='skin_img')
    if ctx['use_hero']:
        add_img_stats(sim_res_ctx,img_type='skin_img')

def add_rotation_stats(res_ctx):
    shared_data_dir=res_ctx['shared_data_dir']
    mean=torch.from_numpy(np.load(join(shared_data_dir,'rotation_mean.npy')))
    res_ctx['rotation_mean']=mean.view(1,-1).to(device=device,dtype=dtype)
    std=torch.from_numpy(np.load(join(shared_data_dir,'rotation_std.npy')))
    std[std<1e-6]=1 # avoid divide by 0
    res_ctx['rotation_std']=std.view(1,-1).to(device=device,dtype=dtype)

if ctx['normalize_rotations']:
    add_rotation_stats(res_ctx)
    if ctx['use_hero']:
        add_rotation_stats(sim_res_ctx)

print('rundir',rundir)
if args.eval!='none':
    print('eval_out_dir',ctx['eval_out_dir'])

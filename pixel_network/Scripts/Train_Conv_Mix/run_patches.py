######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
import argparse
import os
from patch_utils import patchManager

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-run',action='store_true')
    parser.add_argument('-restart',action='store_true')
    parser.add_argument('-vlz',action='store_true')
    parser.add_argument('-eval',action='store_true')

    args=parser.parse_args()

    patch_manager=PatchManager()
    n_gpus=8
    task_per_gpu=4
    n_patches=len(patch_manager.patch_names)

    # python train_conv.py -data_root_dir /data/zhenglin/poses_v3 -show_every_epoch 1 -max_num_samples 64 -batch_size 8
    if args.run:
        for patch_id in range(n_patches):
            gpu_id=patch_id//task_per_gpu+1
            cmd='python train_conv.py -test_case mixres -use_patches -patch_id {} -device cuda:{} -batch_size 16 -use_uvn &'.format(patch_id,gpu_id)
            print(cmd)
            os.system(cmd)

    if args.restart:
        for patch_id in range(n_patches):
            gpu_id=patch_id//task_per_gpu
            patch_name=patch_manager.patch_names[patch_id]
            cmd='python train_conv.py -test_case lowres_patches -patch_id {} -device cuda:{} -batch_size 16 -trial restart -cp ../../rundir/lowres_patches/{}_test/saved_models/model_best.pth.gz &'.format(patch_id,gpu_id,patch_name)
            print(cmd)
            os.system(cmd)

    if args.vlz:
        pairs=[]
        for patch_id in range(n_patches):
            patch_name=patch_manager.patch_names[patch_id]
            # pairs.append("{0}:{0}_test/logs".format(patch_name))
            # pairs.append("{0}:{0}_restart/logs".format(patch_name))
            pairs.append("{0}:{0}_upconv/logs".format(patch_name))
        port=8001
        cmd='tensorboard --port={} --logdir={}'.format(port,','.join(pairs))
        print(cmd)

    if args.eval:
        for patch_id in range(n_patches):
            patch_name=patch_manager.patch_names[patch_id]
            cmd='python train_conv.py -data_root_dir /data/zhenglin/poses_v3 -show_every_epoch 1 -max_num_samples 64 -test_case lowres_normal_patches -trial full_conv_lap -eval test -patch_id {} -cp ../../rundir/lowres_normal_patches/full_conv_lap/{}/saved_models/model_best.pth.tar -use_normals -lambda_normal 0.1 -use_uvn -no_shuffle'.format(patch_id,patch_name)
            print(cmd)
            os.system(cmd)
            # break

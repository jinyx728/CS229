######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import numpy as np
import argparse
import gzip

def calc_offset_img_stats(offset_img_dir,train_samples,prefix='offset_img',use_gzip=False):
    offset_img_sum,offset_img_sum_sqr=None,None
    offset_max,offset_min=None,None
    n_valid_samples=0
    for i in train_samples:
        try:
            offset_img_path=os.path.join(offset_img_dir,'{}_{:08d}.npy'.format(prefix,i))
            if gzip:
                with gzip.open('{}.gz'.format(offset_img_path),'rb') as f:
                    offset_img=np.load(file=f)
            else:
                offset_img=np.load(offset_img_path)
            D=offset_img.shape[2]
            if offset_img_sum is None:
                offset_img_sum=offset_img
                offset_img_sum_sqr=offset_img**2
                offset_max=np.max(offset_img.reshape(-1,D),axis=0)
                offset_min=np.min(offset_img.reshape(-1,D),axis=0)
            else:
                offset_img_sum+=offset_img
                offset_img_sum_sqr+=offset_img**2
                offset_img_max=np.max(offset_img.reshape(-1,D),axis=0)
                offset_max=np.max(np.vstack([offset_max,offset_img_max]),axis=0)
                offset_img_min=np.min(offset_img.reshape(-1,D),axis=0)
                offset_min=np.min(np.vstack([offset_min,offset_img_min]),axis=0)

            n_valid_samples+=1
        except Exception as e:
            print(e)
    print('n_valid_samples',n_valid_samples)
    offset_img_mean=offset_img_sum/n_valid_samples
    offset_img_std=np.sqrt(offset_img_sum_sqr/n_valid_samples-offset_img_mean**2)
    print(offset_img_mean.shape)
    print(offset_img_std.shape)
    print(offset_max.shape) 
    print(offset_min.shape)
    return offset_img_mean,offset_img_std,offset_max,offset_min

def calc_offset_img_stats_alt_np(offset_img_dir,train_samples):
    all_offset_imgs = []
    for i in train_samples:
        try:
            offset_img_path=os.path.join(offset_img_dir,'offset_img_{:08d}.npy'.format(i))
            all_offset_imgs.append(np.load(offset_img_path))
        except Exception as e:
            print(e)
    
    n_valid_samples = len(all_offset_imgs)
    print('n_valid_samples',n_valid_samples)
    all_offset_imgs=np.array(all_offset_imgs)
    offset_img_mean=np.mean(all_offset_imgs, axis=0)
    offset_img_std=np.std(all_offset_imgs, axis=0)
    offset_img_max=np.amax(all_offset_imgs,axis=(0,1,2))
    offset_img_min=np.amin(all_offset_imgs,axis=(0,1,2))
    print(offset_img_mean.shape)
    print(offset_img_std.shape)
    print(offset_img_max.shape)
    print(offset_img_min.shape)
    return offset_img_mean,offset_img_std,offset_max,offset_min

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-offset_img_dir',default='/data/zhenglin/poses_v3/midres_skin_imgs_256')
    parser.add_argument('-prefix',default='skin_img')
    parser.add_argument('-out_dir',default='../../shared_data_midres')
    parser.add_argument('-train_samples',default='/data/zhenglin/poses_v3/sample_lists/midres_train_samples.txt')
    parser.add_argument('-img_size',default=256)
    parser.add_argument('-gzip',action='store_true')

    args=parser.parse_args()
    print('offset_img_dir',args.offset_img_dir,',out_dir',args.out_dir,',train_samples',args.train_samples)
    
    train_samples=np.loadtxt(args.train_samples).astype(int)
    
    offset_img_mean,offset_img_std,offset_max,offset_min=calc_offset_img_stats(args.offset_img_dir,train_samples,prefix=args.prefix,use_gzip=args.gzip)
    
#     offset_img_mean_np,offset_img_std_np,offset_max_np,offset_min_np = calc_offset_img_stats_alt_np(args.offset_img_dir,train_samples) 
#     print('diff mean', np.sum((offset_img_mean-offset_img_mean_np)**2))
#     print('diff std', np.sum((offset_img_std-offset_img_std_np)**2))
#     print('diff max', np.sum((offset_max-offset_max_np)**2))
#     print('diff min', np.sum((offset_min-offset_min_np)**2))
    # offset_img_mean_std = np.concatenate([offset_img_mean,offset_img_std],axis=2)
    # print('offset_img_mean_std',offset_img_mean_std.shape)

    # np.save(os.path.join(args.out_dir,'offset_img_{}_mean_std.npy'.format(args.img_size)),offset_img_mean_std)
    
    np.save(os.path.join(args.out_dir,'{}_{}_mean.npy'.format(args.prefix,args.img_size)),offset_img_mean)
    np.save(os.path.join(args.out_dir,'{}_{}_std.npy'.format(args.prefix,args.img_size)),offset_img_std)
    # np.save(os.path.join(args.out_dir,'offset_img_more_padding_{}_mean_std.npy'.format(args.img_size)),offset_img_mean_std)
    # np.savetxt(os.path.join(args.out_dir,'offset_more_padding_{}_max_min.txt'.format(args.img_size)),np.vstack([offset_max,offset_min]))
    
#     np.save(os.path.join(args.out_dir,'offset_img_64_mean_std.npy'),offset_img_mean_std)
#     np.savetxt(os.path.join(args.out_dir,'offset_64_max_min.txt'),np.vstack([offset_max,offset_min]))
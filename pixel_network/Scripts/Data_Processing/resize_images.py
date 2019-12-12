######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import numpy as np
import cv2
import argparse
import gzip

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('-data_root_dir',default='/data/zhenglin/poses_v3/')
    parser.add_argument('-offset_img_dir',default='midres_uvn_offset_imgs')
    parser.add_argument('-offset_img_prefix',default='offset_img')
    parser.add_argument('-start',type=int,default=0)
    parser.add_argument('-end',type=int,default=14999)
    parser.add_argument('-overwrite',action='store_true')
    parser.add_argument('-output_size',type=int,default=256)
    parser.add_argument('-gzip',action='store_true')
    args=parser.parse_args()

    project_root_dir = '../../..'
    learning_root_dir = os.path.join(project_root_dir, 'Learning2')
    if args.offset_img_dir.find('lowres')!=-1:
        shared_data_dir = os.path.join(learning_root_dir, 'shared_data')
    else:
        shared_data_dir = os.path.join(learning_root_dir, 'shared_data_midres')
    print('shared_data_dir',shared_data_dir)

    data_root_dir = args.data_root_dir

    offset_img_dir = os.path.join(data_root_dir, args.offset_img_dir)
    offset_img_pattern = '{}_{:08d}.npy'
    # num_channels = 6

    mask_file = os.path.join(shared_data_dir, 'offset_img_mask.npy')
    assert(os.path.exists(mask_file))
    mask = np.load(mask_file)

    input_size = (512, 512)

    start_index = args.start
    end_index = args.end


    # output_size = 128
    output_size = args.output_size
    output_img_dir = os.path.join(data_root_dir, '{}_{}'.format(args.offset_img_dir,output_size))
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    # out_mask_file = os.path.join(shared_data_dir, 'offset_img_mask_%d.npy' %output_size)


    def resize(in_arr, out_size, n_channels):
        out_arr = np.empty([out_size, out_size, n_channels], dtype=np.float32)
        for c in range(n_channels):
            out_arr[:, :, c] = cv2.resize(in_arr[:, :, c], (out_size, out_size), interpolation = cv2.INTER_AREA)
        return out_arr

    overwrite = args.overwrite

    # out_mask = resize(mask, output_size, 2)
    # np.save(out_mask_file, out_mask)

    for i in range(start_index, end_index+1):
        try:
            in_file = os.path.join(offset_img_dir, offset_img_pattern.format(args.offset_img_prefix,i))
            out_file = os.path.join(output_img_dir, offset_img_pattern.format(args.offset_img_prefix,i))
            if args.gzip:
                in_file='{}.gz'.format(in_file)
                out_file='{}.gz'.format(out_file)
            
            if os.path.exists(out_file) and not overwrite:
                print(i,'skip')
                continue
            
            if not os.path.exists(in_file):
                print('missing input', in_file)
                continue
            
            print(i)
            # print('in_file',in_file,'out_file',out_file)
            # continue
            
            if args.gzip:
                with gzip.open(in_file,'rb') as f:
                    offsets=np.load(file=f)
            else:
                offsets = np.load(in_file)
            
            resized_offsets = resize(offsets, output_size, offsets.shape[2])
            # print(resized_offsets.shape)

            if args.gzip:
                with gzip.open(out_file,'wb') as f:
                    np.save(file=f,arr=resized_offsets)
            else:
                np.save(out_file, resized_offsets)
            # if (i+1) % 1000 == 0:
            #     print('converted %d samples' %i)
        except Exception as e:
            print(i,'failed')


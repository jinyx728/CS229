######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
import os
import glob
import argparse

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-data_dir',default='/data3/yilinzhu/poses_v3')
    parser.add_argument('-offset_imgs_dir',default='offset_imgs_tie')
    parser.add_argument('-out_dir',default='offset_imgs_crop_tie')
    args=parser.parse_args()

    output_dir = os.path.join(args.data_dir,args.out_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        
    offset_imgs = sorted(glob.glob(os.path.join(args.data_dir,args.offset_imgs_dir,"*.npy")))
    assert(len(offset_imgs))
                 
#     for offset_img in offset_imgs:
#         a = np.load(offset_img)
#         crop_a = a[:,192:320]
#         save_name = os.path.basename(offset_img)
#         print(save_name)
#         np.save(os.path.join(output_dir,save_name),crop_a)
        
    mask_file = '../../shared_data_necktie/offset_img_mask.npy'
    if os.path.isfile(mask_file):
        a = np.load(mask_file)
        crop_a = a[:,192:320]
        save_name = mask_file.replace("img_mask","img_mask_crop")
        print(save_name)
        np.save(save_name,crop_a)
        
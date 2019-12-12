######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool
from contextlib import closing


parser=argparse.ArgumentParser()
parser.add_argument('start',type=int)
parser.add_argument('end',type=int)
parser.add_argument('offset_img_dir',default='lowres_offset_{}_imgs')
parser.add_argument('data_root_dir', type=str)
parser.add_argument('-offset_img_prefix',default='offset_img')
parser.add_argument('-overwrite',action='store_true')
parser.add_argument('-output_size',type=int,default=128)
parser.add_argument("-ncore",type=int,default=10)
args=parser.parse_args()

project_root_dir = '../../..'
learning_root_dir = os.path.join(project_root_dir, 'Learning')
# shared_data_dir = os.path.join(learning_root_dir, 'shared_data')
shared_data_dir = os.path.join(learning_root_dir, 'shared_data')

data_root_dir = args.data_root_dir

offset_img_dir = os.path.join(data_root_dir, args.offset_img_dir)
offset_img_pattern = '{}_{:08d}.npy'
num_channels = 6

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

out_mask_file = os.path.join(shared_data_dir, 'offset_img_mask_%d.npy' %output_size)


def resize(in_arr, out_size, n_channels):
    out_arr = np.empty([out_size, out_size, n_channels], dtype=np.float32)
    for c in range(n_channels):
        out_arr[:, :, c] = cv2.resize(in_arr[:, :, c], (out_size, out_size), interpolation = cv2.INTER_AREA)
    return out_arr

overwrite = args.overwrite

out_mask = resize(mask, output_size, 2)
np.save(out_mask_file, out_mask)

def f(i):
    in_file = os.path.join(offset_img_dir, offset_img_pattern.format(args.offset_img_prefix,i))
    out_file = os.path.join(output_img_dir, offset_img_pattern.format(args.offset_img_prefix,i))

    if os.path.exists(out_file) and not overwrite:
        return
    if not os.path.exists(in_file):
        print(in_file+" not exist")
        return
    
    offsets = np.load(in_file)
    resized_offsets = resize(offsets, output_size, num_channels)
    np.save(out_file, resized_offsets)
    if (i+1) % 100 == 0:
        print('converted {} samples'.format(i+1))


print("{} cores are running.".format(args.ncore))
with closing(Pool(args.ncore)) as pool:
    pool.map(f, range(start_index, end_index+1), chunksize=None)
    pool.terminate()

######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
import os
from rotation_utils import read_quaternions_from_file, quaternion_to_rotation_matrix
import argparse

parser = argparse.ArgumentParser(description="convert quat to matrices")
parser.add_argument("start", type=int, 
    help="start index (included)")
parser.add_argument("end", type=int, 
    help="end index (included)")
parser.add_argument("dir", type=str, 
    help="body pose dir")
parser.add_argument("-mod", type=str, 
    default=None)

args = parser.parse_args()

start_index = args.start
end_index = args.end

project_root_dir = '../../..'
learning_root_dir = os.path.join(project_root_dir, 'Learning')

# input quaternions data
data_root_dir = args.dir
if args.mod:
    quaternions_dir = os.path.join(data_root_dir, 'rotations_{}'.format(args.mod))
else:
    quaternions_dir = os.path.join(data_root_dir, 'rotations')

quaternion_pattern = 'rotation_%08d.txt'
num_joints = 10


# output rotation matrices data
if args.mod:
    rotation_mats_dir = os.path.join(data_root_dir, 'rotation_matrices_{}'.format(args.mod))
else:   
    rotation_mats_dir = os.path.join(data_root_dir, 'rotation_matrices')
if not os.path.exists(rotation_mats_dir):
    os.makedirs(rotation_mats_dir)
rotation_mats_pattern = 'rotation_mat_%08d.npy'

cnt = 0
for i in range(start_index, end_index+1):
    if i % 1000 == 0:
        print('convert rotation matrix {:08d}'.format(i))
    quaternion_filename =os.path.join(quaternions_dir, quaternion_pattern %i)
    if not os.path.exists(quaternion_filename):
        print(quaternion_filename+" not exist. skipping.")
        continue
    quaternions = read_quaternions_from_file(quaternion_filename, num_joints=num_joints)
    rotation_mats = np.empty([num_joints, 9])
    for j in range(num_joints):
        rotation_mats[j, :] = quaternion_to_rotation_matrix(quaternions[j, :]).reshape(-1)
    #print(rotation_mats)
    rotation_mats_filename = os.path.join(rotation_mats_dir, rotation_mats_pattern %i)
    np.save(rotation_mats_filename, rotation_mats)
    cnt += 1
print('converted %d poses' %cnt)
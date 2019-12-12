######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
import os
import subprocess
from obj_io import Obj, read_obj, write_obj
import argparse

parser = argparse.ArgumentParser(description="compute offset")
parser.add_argument("start", type=int, 
    help="start index (included)")
parser.add_argument("end", type=int, 
    help="end index (included)")
parser.add_argument("body_shape", type=str, 
    help="body shape keyword")
parser.add_argument("dir", type=str, 
    help="body pose dir")
parser.add_argument("project_dir", type=str, 
    help="cloth project dir")


args = parser.parse_args()


print("## set directories, paths, and prefixes/suffixes")

project_root_dir = '../../..'
learning_root_dir = os.path.join(project_root_dir, 'Learning')
shared_data_dir = os.path.join(learning_root_dir, 'shared_data')

# input sim data
data_root_dir = args.dir

resolution = 'midres'

tshirt_dir = os.path.join(data_root_dir, resolution + '_tshirts')
skin_tshirt_dir = os.path.join(data_root_dir, resolution + '_skin_tshirts_{}'.format(args.body_shape))
tshirt_prefix = 'tshirt'
skin_tshirt_prefix = 'skin_tshirt'
tshirt_pattern = '%s_%08d'

start_index = args.start
end_index = args.end

# output npy data
skin_npy_dir = os.path.join(data_root_dir, resolution + '_skin_{}_npys'.format(args.body_shape))
if not os.path.exists(skin_npy_dir):
    os.makedirs(skin_npy_dir)
skin_npy_pattern = 'skin_%08d.npy'

tshirt_npy_dir = os.path.join(data_root_dir, resolution + '_tshirt_npys')
if not os.path.exists(tshirt_npy_dir):
    os.makedirs(tshirt_npy_dir)
tshirt_npy_pattern = 'tshirt_%08d.npy'

offset_npy_dir = os.path.join(data_root_dir, resolution + '_offset_{}_npys'.format(args.body_shape))
if not os.path.exists(offset_npy_dir):
    os.makedirs(offset_npy_dir)
offset_npy_pattern = 'offset_%08d.npy'


print("## get faces of rest flat t-shirt")

tshirt_faces_filename = os.path.join(shared_data_dir, 'tshirt_faces.npy')
rest_flat_obj_filename = os.path.join(shared_data_dir, 'flat_tshirt.obj')
rest_flat_obj = read_obj(rest_flat_obj_filename)
np.save(tshirt_faces_filename, rest_flat_obj.f)
print('wrote tshirt faces to file', tshirt_faces_filename)


print("## convert tri to obj")

for i in range(start_index, end_index+1):
    # convert draped tshirt
    draped_tshirt_obj_filename = os.path.join(tshirt_dir, tshirt_pattern %(tshirt_prefix, i) +'.obj')
    if not os.path.exists(draped_tshirt_obj_filename):
        draped_tshirt_tri_filename = os.path.join(tshirt_dir, tshirt_pattern %(tshirt_prefix, i) + '.tri.gz')
        if os.path.exists(draped_tshirt_tri_filename):
            tri2obj_command = '$PHYSBAM/Tools/tri2obj/tri2obj %s %s' %(draped_tshirt_tri_filename, draped_tshirt_obj_filename)
            subprocess.call(tri2obj_command, shell=True)
    # convert skin tshirt
    skin_tshirt_obj_filename = os.path.join(skin_tshirt_dir, tshirt_pattern %(skin_tshirt_prefix, i) + '.obj')
    if not os.path.exists(skin_tshirt_obj_filename):
        skin_tshirt_tri_filename = os.path.join(skin_tshirt_dir, tshirt_pattern %(skin_tshirt_prefix, i) + '.tri.gz')
        if os.path.exists(skin_tshirt_tri_filename):
            tri2obj_command = '$PHYSBAM/Tools/tri2obj/tri2obj %s %s' %(skin_tshirt_tri_filename, skin_tshirt_obj_filename)
            subprocess.call(tri2obj_command, shell=True)
            

print("## save skin t-shirt vertices to npy")
            
cnt = 0
for i in range(start_index, end_index+1):
    skin_filename = os.path.join(skin_npy_dir, skin_npy_pattern %i)
    if os.path.exists(skin_filename):
        continue
    # load skin tshirt
    skin_tshirt_obj_filename = os.path.join(skin_tshirt_dir, tshirt_pattern %(skin_tshirt_prefix, i) + '.obj')
    skin_tshirt_obj = read_obj(skin_tshirt_obj_filename)
    if skin_tshirt_obj is None:
        print('missing %s' % skin_tshirt_obj_filename)
        continue
    else:
        cnt += 1
        np.save(skin_filename, skin_tshirt_obj.v)
print('converted %d to npy' % cnt)



print("## save draped t-shirt vertices to npy")

cnt = 0
for i in range(start_index, end_index+1):
    draped_filename = os.path.join(tshirt_npy_dir, tshirt_npy_pattern %i)
    if os.path.exists(draped_filename):
        continue
    # load draped tshirt
    draped_tshirt_obj_filename = os.path.join(tshirt_dir, tshirt_pattern %(tshirt_prefix, i) + '.obj')
    draped_tshirt_obj = read_obj(draped_tshirt_obj_filename)
    if draped_tshirt_obj is None:
        print('missing %s' %draped_tshirt_obj_filename)
        continue
    else:
        cnt += 1
        np.save(draped_filename, draped_tshirt_obj.v)
print('converted %d to npy' %cnt)


print("## compute and save offsets to npy")

cnt = 0
for i in range(start_index, end_index+1):
    if i%500 == 0:
        print("computed and saved {} offsets to npy. {}".format(i, args.body_shape))
    # load draped tshirt
    tshirt_npy_filename = os.path.join(tshirt_npy_dir, tshirt_npy_pattern %i)
    if not os.path.exists(tshirt_npy_filename):
        continue
    draped_verts = np.load(tshirt_npy_filename)
    # load skin tshirt
    skin_npy_filename = os.path.join(skin_npy_dir, skin_npy_pattern %i)
    skin_verts = np.load(skin_npy_filename)
    assert(skin_verts.shape == draped_verts.shape)
    # compute offset
    v_diff = draped_verts - skin_verts
    # save offset
    offset_npy_filename = os.path.join(offset_npy_dir, offset_npy_pattern %i)
    np.save(offset_npy_filename, v_diff)
    cnt += 1
print('saved %d offsets to npy' %cnt)
import bpy
import os
from os.path import isfile,isdir,join
from io_utils import read_vt

def set_vt(obj,vt):
    n=len(vt)
    for i in range(n):
        obj.data.uv_layers[0].data[i].uv=vt[i]

def load_cam(path):
    with open(path) as f:
        lines=f.readlines()
        parts=lines[0].split()
        x=[float(parts[0]),float(parts[1]),float(parts[2])]
        parts=lines[1].split()
        q=[float(parts[0]),float(parts[1]),float(parts[2]),float(parts[3])]
    return x,q

def set_blender_frames(in_dir):
    with open(join(in_dir,'last_frame.txt')) as f:
        line=f.readline()
        parts=line.split()
        n_key_frames,n_total_frames=int(parts[0]),int(parts[1])
    cam=bpy.data.objects['Camera']

    for frame_i in range(n_total_frames):
        print('set frame',frame_i)
        bpy.context.scene.frame_set(frame_i+1)
        cam_path=join(in_dir,'cam_{}.txt'.format(frame_i))
        x,q=load_cam(cam_path)
        cam.location=x
        cam.rotation_quaternion=q
        cam.keyframe_insert(data_path='location',frame=frame_i+1)
        cam.keyframe_insert(data_path='rotation_quaternion',frame=frame_i+1)


# set_blender_frames('camera_test/5983/blender_square_120')
set_blender_frames('reconstruct_test/00016107_square/blender_cam')

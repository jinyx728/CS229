######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Data_Processing')
import numpy as np
from bone_utils import BoneStructure, Bone
import os
from obj_io import Obj, read_obj, write_obj

def load_skin_weights(skin_weights_path):
    bone_structure=BoneStructure()
    mesh_weights=[]
    world_from_blender=np.zeros((4,4))
    world_from_blender[0,0]=1
    world_from_blender[1,2]=1
    world_from_blender[2,1]=-1
    world_from_blender[3,3]=1

    # world_from_blender[0,0]=1
    # world_from_blender[1,1]=1
    # world_from_blender[2,2]=1
    # world_from_blender[3,3]=1

    with open(skin_weights_path) as f:
        line=f.readline()
        n_bones=int(line)
        for bone_id in range(n_bones):
            line=f.readline()
            parts=line.split()
            bone_name=parts[0]
            parent_name=parts[1]
            if parent_name!='None':
                parent=bone_structure.bone_dict[parent_name]
            else:
                parent=None
            rest_world_frame=np.array([float(parts[i]) for i in range(2,18)]).reshape((4,4))
            world_tail=np.array([float(parts[i]) for i in range(18,21)])
            world_head=rest_world_frame[:3,3]
            local_tail=np.linalg.inv(rest_world_frame[:3,:3]).dot(world_tail-world_head)
            for i in range(3):
                rest_world_frame[i,:]/=np.linalg.norm(rest_world_frame[i,:3])
            rest_world_frame=world_from_blender.dot(rest_world_frame)
            bone=Bone(bone_id,bone_name,parent,rest_world_frame,local_tail)
            bone_structure.add_bone(bone)

        line=f.readline()
        n_vts=int(line)
        for vt_id in range(n_vts):
            line=f.readline()
            parts=line.split()
            bone_ids=[]
            weights=[]
            sum_weight=0
            for part in parts:
                bone_weight=part.split(':')
                bone_id=int(bone_weight[0])
                weight=float(bone_weight[1])
                bone_ids.append(bone_id)
                weights.append(weight)
                sum_weight+=weight

            for i in range(len(weights)):
                weights[i]/=sum_weight

            vertex_weights=[]
            for i in range(len(weights)):
                vertex_weights.append((bone_ids[i],weights[i]))

            mesh_weights.append(vertex_weights)

    return bone_structure,mesh_weights

def load_cloth_embeddings(embedding_path):
    vertex_embeds=[]
    with open(embedding_path) as f:
        while True:
            line=f.readline()
            if line=='':
                break
            parts=line.split()
            face_id=int(parts[0])-1
            weights=(float(parts[1]),float(parts[2]),float(parts[3]))
            vertex_embeds.append((face_id,weights))
    return vertex_embeds

def get_skin_cloth_vts(body_obj,cloth_embeds):
    vts=[]
    for face_id,weights in cloth_embeds:
        fc=body_obj.f[face_id]
        x0,x1,x2=body_obj.v[fc[0]],body_obj.v[fc[1]],body_obj.v[fc[2]]
        vts.append(x0*weights[0]+x1*weights[1]+x2*weights[2])
    vts=np.array(vts)
    return vts

if __name__=='__main__':
    shared_data_dir='../../shared_data'
    embedding_path=os.path.join(shared_data_dir,'dressed_TshirtW_embedding.txt')
    model_path=os.path.join(shared_data_dir,'modelT.obj')
    modelT=read_obj(model_path)
    
    vertex_embeds=load_cloth_embeddings(embedding_path)
    vts=np.array(modelT,vertex_embeds)
    obj=Obj(v=vts,f=np.empty((0,3)))
    write_obj(obj,'cloth_skin.obj')

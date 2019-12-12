######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import numpy as np
from skin_utils import load_skin_weights, load_cloth_embeddings, get_skin_cloth_vts
from obj_io import Obj, write_obj, read_obj
from collections import defaultdict
# from multiprocessing.tool import ThreadPool
# import copy

def load_bone_names(path):
    bone_names=[]
    with open(path) as f:
        while True:
            line=f.readline()
            if line is '':
                break
            bone_name=line.rstrip()
            bone_names.append(bone_name)
    return bone_names

def load_rotation_dict(rotation_file_path,bone_names):
    rotation_dict={}
    rotations=np.load(rotation_file_path)
    assert(len(bone_names)==len(rotations))
    for i in range(len(bone_names)):
        bone_name=bone_names[i]
        rotation=rotations[i].reshape(3,3)
        rotation_dict[bone_name]=rotation
    return rotation_dict

def write_bone_origins_obj(bone_structure,out_file):
    vts=[]
    for bone in bone_structure.bone_list:
        vts.append(bone.world_frame[:3,3])
    vts=np.array(vts)
    fcs=np.empty((0,3))
    obj=Obj(v=vts,f=fcs)
    write_obj(obj,out_file)

def get_cloth_bone_weights(body_bone_weights_list,cloth_body_weights):
    assert(len(body_bone_weights_list)==len(cloth_body_weights))
    cloth_bone_weights_dict=defaultdict(float)
    for i in range(len(body_bone_weights_list)):
        body_bone_weights=body_bone_weights_list[i]
        cloth_body_weight=cloth_body_weights[i]
        for bone_id,body_bone_weight in body_bone_weights:
            cloth_bone_weights_dict[bone_id]+=cloth_body_weight*body_bone_weight
    return cloth_bone_weights_dict.items()

def get_whole_cloth_bone_weights(whole_cloth_body_weights,whole_body_bone_weights,body_fcs):
    n_cloth_vts=len(whole_cloth_body_weights)
    whole_cloth_bone_weights=[]
    for cloth_vt_id in range(n_cloth_vts):
        body_fc_id,cloth_body_weights=whole_cloth_body_weights[cloth_vt_id]
        body_fc=body_fcs[body_fc_id]
        body_bone_weights_list=[whole_body_bone_weights[body_fc[0]],whole_body_bone_weights[body_fc[1]],whole_body_bone_weights[body_fc[2]]]
        cloth_bone_weights=get_cloth_bone_weights(body_bone_weights_list,cloth_body_weights)
        whole_cloth_bone_weights.append(cloth_bone_weights)
    return whole_cloth_bone_weights

def transform_cloth_offsets(cloth_offsets,bone_structure,whole_cloth_bone_weights):
    cloth_offsets_T=[]
    for cloth_vt_id in range(len(cloth_offsets)):
        cloth_vt_offset=cloth_offsets[cloth_vt_id]
        cloth_bone_weights=whole_cloth_bone_weights[cloth_vt_id]
        R=np.zeros((3,3))

        for bone_id,bone_weight in cloth_bone_weights:
            bone=bone_structure.bone_list[bone_id]
            R+=bone.world_frame[:3,:3].dot(bone.rest_world_frame[:3,:3].T)*bone_weight
            # cloth_T_vt_offset+=bone.rest_world_frame[:3,:3].dot(bone.world_frame[:3,:3].T).dot(cloth_vt_offset)*bone_weight
        cloth_vt_offset_T=np.linalg.inv(R).dot(cloth_vt_offset)
        cloth_offsets_T.append(cloth_vt_offset_T)
    return np.array(cloth_offsets_T)

def transform_cloth_offsets_T(cloth_offsets_T,bone_structure,whole_cloth_bone_weights):
    cloth_offsets=[]
    for cloth_vt_id in range(len(cloth_offsets_T)):
        cloth_vt_offset_T=cloth_offsets_T[cloth_vt_id]
        cloth_bone_weights=whole_cloth_bone_weights[cloth_vt_id]
        R=np.zeros((3,3))

        for bone_id,bone_weight in cloth_bone_weights:
            bone=bone_structure.bone_list[bone_id]
            R+=bone.world_frame[:3,:3].dot(bone.rest_world_frame[:3,:3].T)*bone_weight

        cloth_vt_offset=R.dot(cloth_vt_offset_T)
        cloth_offsets.append(cloth_vt_offset)

    return np.array(cloth_offsets)

def write_line_obj(path,v,l):
    with open(path,'w') as f:
        for vi in v:
            f.write('v {} {} {}\n'.format(vi[0],vi[1],vi[2]))
        for li in l:
            f.write('l {} {}\n'.format(li[0]+1,li[1]+1))

class RotationTransformer:
    def __init__(self):
        learning_root_dir='../../'
        self.shared_data_dir=os.path.join(learning_root_dir,'shared_data')
        self.data_root_dir='/data/zhenglin/poses_v3'
        self.rotation_dir=os.path.join(self.data_root_dir,'rotation_matrices')
        self.world_rotation_dir=os.path.join(self.data_root_dir,'world_rotation_matrices')
        if not os.path.isdir(self.world_rotation_dir):
            os.mkdir(self.world_rotation_dir)

        bone_names_path=os.path.join(self.shared_data_dir,'rotation_readme.txt')
        self.bone_names=load_bone_names(bone_names_path)
        skin_weights_path=os.path.join(self.shared_data_dir,'skin_weights.txt')
        self.bone_structure,self.whole_body_bone_weights=load_skin_weights(skin_weights_path)

    def read_rotation_dict_from_id(self,test_id):
        rotations_path=os.path.join(self.rotation_dir,'rotation_mat_{:08d}.npy'.format(test_id))
        rotation_dict=load_rotation_dict(rotations_path,self.bone_names)
        return rotation_dict

    def write_world_rotation(self,test_id):
        rotation_dict=self.read_rotation_dict_from_id(test_id)
        self.bone_structure.apply_local_rotations(rotation_dict)
        world_rotations=[]
        for bone_name in self.bone_names:
            bone=self.bone_structure.bone_dict[bone_name]
            world_rotations.append(bone.world_frame[:3,:3].reshape(-1))
        world_rotations=np.array(world_rotations)
        world_rotation_path=os.path.join(self.world_rotation_dir,'rotation_mat_{:08d}.npy'.format(test_id))
        np.save(world_rotation_path,world_rotations)

    def write_bone_obj(self,test_id):
        rotation_dict=self.read_rotation_dict_from_id(test_id)
        self.bone_structure.apply_local_rotations(rotation_dict)
        # import pyquaternion
        # m=pyquaternion.Quaternion([0,1,0,0]).rotation_matrix
        # rotation_dict={'LeftArm':m}
        # self.bone_structure.apply_local_rotations(rotation_dict)
        v,l=self.bone_structure.get_bone_obj()
        obj_path='body_test/bone_{:08d}.obj'.format(test_id)
        print('write to',obj_path)
        write_line_obj(obj_path,v*1e-1,l)

class ClothTransformer:
    def __init__(self):
        learning_root_dir='../../'
        self.shared_data_dir=os.path.join(learning_root_dir,'shared_data')

        # self.data_root_dir='/data/njin19/generated_models_v2'
        self.data_root_dir='/data/zhenglin/poses_v3'
        self.offsets_dir=os.path.join(self.data_root_dir,'lowres_offset_npys')
        self.offsets_T_dir='/data/zhenglin/poses_v3/lowres_offset_T_npys'
        self.skin_dir=os.path.join(self.data_root_dir,'lowres_skin_npys')
        self.rotation_dir=os.path.join(self.data_root_dir,'rotation_matrices')

        bone_names_path=os.path.join(self.shared_data_dir,'rotation_readme.txt')
        self.bone_names=load_bone_names(bone_names_path)

        skin_weights_path=os.path.join(self.shared_data_dir,'skin_weights.txt')
        self.bone_structure,self.whole_body_bone_weights=load_skin_weights(skin_weights_path)

        modelT_path=os.path.join(self.shared_data_dir,'modelT.obj')
        modelT=read_obj(modelT_path)
        self.body_fcs=modelT.f

        cloth_embeddings_path=os.path.join(self.shared_data_dir,'dressed_TshirtW_embedding.txt')
        self.whole_cloth_body_weights=load_cloth_embeddings(cloth_embeddings_path)
        self.skin_cloth_T_vts=get_skin_cloth_vts(modelT,self.whole_cloth_body_weights)

        # flat_tshirt_path=os.path.join(self.shared_data_dir,'flat_tshirt.obj')
        flat_tshirt_path=os.path.join(self.shared_data_dir,'flat_TshirtW_remesh3_lowres_tri.obj')
        flat_tshirt_obj=read_obj(flat_tshirt_path)
        self.tshirt_fcs=flat_tshirt_obj.f

        self.whole_cloth_bone_weights=get_whole_cloth_bone_weights(self.whole_cloth_body_weights,self.whole_body_bone_weights,self.body_fcs)

    def read_rotation_dict_from_id(self,test_id):
        rotations_path=os.path.join(self.rotation_dir,'rotation_mat_{:08d}.npy'.format(test_id))
        rotation_dict=load_rotation_dict(rotations_path,self.bone_names)
        return rotation_dict

    def read_offsets_from_id(self,test_id,data_dir=None,pattern=None):
        if data_dir is None:
            data_dir=self.offsets_dir
            pattern='offset_{:08d}.npy'
        offsets_path=os.path.join(data_dir,pattern.format(test_id))
        return np.load(offsets_path)

    def read_offsets_T_from_id(self,test_id,data_dir=None,pattern=None):
        if data_dir is None:
            data_dir=self.offsets_T_dir
            pattern='offsets_T_{:08d}.npy'
        offsets_path=os.path.join(data_dir,pattern.format(test_id))
        return np.load(offsets_path)


    def get_offsets_T_from_id(self,test_id,write_offsets=False):
        rotations_path=os.path.join(self.rotation_dir,'rotation_mat_{:08d}.npy'.format(test_id))
        rotation_dict=load_rotation_dict(rotations_path,self.bone_names)

        offsets_path=os.path.join(self.offsets_dir,'offset_{:08d}.npy'.format(test_id))
        offsets=np.load(offsets_path)

        offsets_T=self.get_offsets_T(offsets,rotation_dict)

        if write_offsets:
            out_path=os.path.join(self.offsets_T_dir,'offsets_T_{:08d}.npy'.format(test_id))
            np.save(out_path,offsets_T)

        return offsets_T

    def get_offsets_from_id(self,test_id):
        rotations_path=os.path.join(self.rotation_dir,'rotation_mat_{:08d}.npy'.format(test_id))
        rotation_dict=load_rotation_dict(rotations_path,self.bone_names)

        offsets_T_path=os.path.join(self.offsets_T_dir,'offsets_T_{:08d}.npy'.format(test_id))
        offsets_T=np.load(offsets_T_path)

        offsets=self.get_offsets(offsets_T,rotation_dict)

        return offsets

    def get_offsets_T(self,offsets,rotation_dict):
        self.bone_structure.apply_local_rotations(rotation_dict)
        offsets_T=transform_cloth_offsets(offsets,self.bone_structure,self.whole_cloth_bone_weights)
        return offsets_T

    def get_offsets(self,offsets_T,rotation_dict):
        self.bone_structure.apply_local_rotations(rotation_dict)
        offsets=transform_cloth_offsets_T(offsets_T,self.bone_structure,self.whole_cloth_bone_weights)
        return offsets

    def write_cloth_T(self,test_id,offsets_T,prefix='',out_dir=''):
        cloth_T_obj=Obj(v=offsets_T+self.skin_cloth_T_vts,f=self.tshirt_fcs)
        cloth_T_path=os.path.join(out_dir,'{}clothT_{:08d}.obj'.format(prefix,test_id))
        write_obj(cloth_T_obj,cloth_T_path)

    def write_cloth(self,test_id,offsets,prefix='',out_dir=''):
        skin_path=os.path.join(self.skin_dir,'skin_{:08d}.npy'.format(test_id))
        skin=np.load(skin_path)
        cloth_obj=Obj(v=offsets+skin,f=self.tshirt_fcs)
        cloth_path=os.path.join(out_dir,'{}cloth_{:08d}.obj'.format(prefix,test_id))
        write_obj(cloth_obj,cloth_path)


if __name__=='__main__':
    # learning_root_dir='../../'
    # shared_data_dir=os.path.join(learning_root_dir,'shared_data')
    # bone_names_path=os.path.join(shared_data_dir,'rotation_readme.txt')
    # skin_weights_path=os.path.join(shared_data_dir,'skin_weights.txt')

    # data_root_dir='/data/njin19/generated_models_v2'
    # test_id=5000
    # rotations_path=os.path.join(data_root_dir,'rotation_matrices/rotation_mat_{:08d}.npy'.format(test_id))

    # bone_names=load_bone_names(bone_names_path)
    # print('# bones',len(bone_names))
    # rotation_dict=load_rotation_dict(rotations_path,bone_names)
    # bone_structure,whole_body_bone_weights=load_skin_weights(skin_weights_path)
    # bone_structure.apply_local_rotations(rotation_dict)

    # offsets_path=os.path.join(data_root_dir,'offset_npys/offset_{:08d}.npy'.format(test_id))
    # offsets=np.load(offsets_path)
    # skin_path=os.path.join(data_root_dir,'skin_npys/skin_{:08d}.npy'.format(test_id))
    # skin=np.load(skin_path)

    # modelT_path=os.path.join(shared_data_dir,'modelT.obj')
    # modelT=read_obj(modelT_path)
    # body_fcs=modelT.f

    # cloth_embeddings_path=os.path.join(shared_data_dir,'dressed_TshirtW_embedding.txt')
    # whole_cloth_body_weights=load_cloth_embeddings(cloth_embeddings_path)

    # skin_cloth_T_vts=get_skin_cloth_vts(modelT,whole_cloth_body_weights)
    
    # ct=ClothTransformer()
    # # offsets_T=transform_cloth_offsets(offsets,bone_structure,whole_body_bone_weights,whole_cloth_body_weights,body_fcs)
    # offsets_T=ct.get_offsetsT(test_id)

    # flat_tshirt_path=os.path.join(shared_data_dir,'flat_tshirt.obj')
    # flat_tshirt_obj=read_obj(flat_tshirt_path)
    # cloth_T_obj=Obj(v=offsets_T+skin_cloth_T_vts,f=flat_tshirt_obj.f)
    # cloth_T_path='clothT_{:08d}.obj'.format(test_id)
    # write_obj(cloth_T_obj,cloth_T_path)
    # write_bone_origins_obj(bone_structure,'bone_origins_{:08d}.obj'.format(test_id))

    # transformer=ClothTransformer()
    # for i in range(15000):
    #     print(i)
    #     try:
    #         offsets_T=transformer.get_offsets_T_from_id(i,write_offsets=True)
    #     except Exception as e:
    #         print(str(e))
         # transformer.write_cloth_T(i,offsets_T)

    # test_id=5000

    # rotations_path=os.path.join(transformer.data_root_dir,'rotation_matrices/rotation_mat_{:08d}.npy'.format(test_id))
    # if not os.path.isfile(rotations_path):
    #     print('cannot find',rotations_path)

    # rotation_dict=load_rotation_dict(rotations_path,transformer.bone_names)

    # offsets_path=os.path.join(transformer.data_root_dir,'offset_npys/offset_{:08d}.npy'.format(test_id))
    # if not os.path.isfile(offsets_path):
    #     print('cannot find',offsets_path)
    # offsets=np.load(offsets_path)
    # print('offsets',offsets)

    # offsets_T=transformer.get_offsets_T(offsets,rotation_dict)
    # new_offsets=transformer.get_offsets(offsets_T,rotation_dict)
    # print('new_offsets',new_offsets)

    # print('diff',np.linalg.norm(offsets-new_offsets))

    # offsets_T=transformer.get_offsets_from_id(test_id)
    # offsets=transformer.get_offsets_from_id(test_id)
    # transformer.write_cloth(test_id,offsets)

    # test_id=2719
    # out_dir='{:08d}'.format(test_id)
    # if not os.path.isdir(out_dir):
    #     os.mkdir(out_dir)

    # offsets=transformer.read_offsets_from_id(test_id)
    # transformer.write_cloth(test_id,offsets,prefix='gt_',out_dir=out_dir)

    # offsets_T=transformer.read_offsets_T_from_id(test_id)
    # transformer.write_cloth_T(test_id,offsets_T,prefix='gt_',out_dir=out_dir)

    # pred_offsets_dir='/data/zhenglin/generated_models/pred_relu_l2/train/whole'
    # pred_offsets_dir='/data/zhenglin/generated_models/pred_relu_l2/test/whole'
    # pred_offsets_path=os.path.join(pred_offsets_dir,'pred_offset_{:08d}.npy'.format(test_id))
    # pred_offsets_T=np.load(pred_offsets_path)
    # transformer.write_cloth_T(test_id,pred_offsets_T,prefix='pred_l2_',out_dir=out_dir)

    # rotation_dict=transformer.read_rotation_dict_from_id(test_id)
    # offsets_T=transformer.get_offsets_T(offsets,rotation_dict)
    # transformer.write_cloth_T(test_id,offsets_T,prefix='gt_',out_dir=out_dir)

    # pred_offsets=transformer.get_offsets(pred_offsets_T,rotation_dict)
    # transformer.write_cloth(test_id,pred_offsets,prefix='pred_l2_',out_dir=out_dir)

    # print('loss',np.sum((offsets_T-pred_offsets_T)**2)/offsets_T.shape[0]/offsets_T.shape[1])

    transformer=RotationTransformer()
    # for i in range(15000):
    #     print(i)
    #     # transformer.write_world_rotation(i)
    #     try:
    #         transformer.write_world_rotation(i)
    #     except Exception as e:
    #         print(e)

    transformer.write_bone_obj(106)




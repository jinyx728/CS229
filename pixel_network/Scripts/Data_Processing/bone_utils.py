######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np

class Bone:
    def __init__(self,id,name,parent,rest_world_frame,local_tail):
        self.id=id
        self.parent=parent
        self.name=name
        self.set_rest_world_frame(rest_world_frame)
        self.set_world_frame(rest_world_frame)
        self.local_tail=local_tail

    def set_world_frame(self,frame):
        self.world_frame=frame
        if self.parent is not None:
            self.local_frame=np.linalg.inv(self.parent.world_frame).dot(self.world_frame)
        else:
            self.local_frame=frame

    def set_local_frame(self,frame):
        self.local_frame=frame
        if self.parent is not None:
            self.world_frame=self.parent.world_frame.dot(frame)
        else:
            self.world_frame=frame

    def update_world_frame(self):
        '''
        used when parent.world_frame is changed
        '''
        if self.parent is not None:
            self.world_frame=self.parent.world_frame.dot(self.local_frame)

    def set_rest_world_frame(self,frame):
        self.rest_world_frame=frame
        if self.parent is not None:
            self.rest_local_frame=np.linalg.inv(self.parent.rest_world_frame).dot(frame)
        else:
            self.rest_local_frame=frame

    def get_rest_world_tail(self):
        return self.rest_world_frame[:3,:3].dot(self.local_tail)+self.rest_world_frame[:3,3]

class BoneStructure:
    def __init__(self):
        self.bone_dict={}
        self.bone_list=[]

    def add_bone(self,bone):
        self.bone_dict[bone.name]=bone
        self.bone_list.append(bone)

    def reset(self):
        for bone in self.bone_list:
            bone.world_frame=bone.rest_world_frame
            bone.local_frame=bone.rest_local_frame

    def apply_local_rotations(self,rotations_dict):
        for bone in self.bone_list:
            if bone.name in rotations_dict:
                local_rotation=rotations_dict[bone.name]
                local_frame=bone.local_frame.copy()
                # local_frame[:3,:3]=bone.rest_local_frame[:3,:3].dot(local_rotation)
                # local_frame[:3,:3]=local_rotation.dot(bone.rest_local_frame[:3,:3])
                R=bone.rest_local_frame[:3,:3]
                # local_rotation=R.dot(local_rotation).dot(R.T)
                # local_frame[:3,:3]=local_rotation.dot(bone.rest_local_frame[:3,:3])
                local_frame[:3,:3]=R.dot(local_rotation)
                # local_frame[:3,:3]=local_rotation[:3,:3]
                # print('apply_local_rotations:name',bone.name,'det:',np.linalg.det(local_rotation))
                # local_frame[:3,:3]=local_rotation.dot(local_frame[:3,:3])
                bone.set_local_frame(local_frame)
            else:
                bone.update_world_frame()

    def apply_world_rotations(self,rotations_dict):
        for bone in self.bone_list:
            if bone.name in rotations_dict:
                local_t=bone.rest_local_frame[:3,3]
                world_t=bone.parent.world_frame[:3,:3].dot(local_t)+bone.parent.world_frame[:3,3]
                world_frame=bone.world_frame.copy()
                world_frame[:3,:3]=rotations_dict[bone.name]
                world_frame[:3,3]=world_t
                bone.set_world_frame(world_frame)
            else:
                bone.update_world_frame()

    def get_bone_obj(self):
        v=[]
        l=[]
        n_vts=0
        for bone in self.bone_list:
            head=bone.world_frame[:3,3]
            tail=bone.world_frame[:3,:3].dot(bone.local_tail)+head
            v+=[head,tail]
            # v.append(bone.world_frame[:3,3])
            # if bone.parent is not None:
            #     l.append([bone.parent.id,bone.id])
            l+=[[n_vts,n_vts+1]]
            n_vts+=2
        return np.array(v),np.array(l).astype(int)


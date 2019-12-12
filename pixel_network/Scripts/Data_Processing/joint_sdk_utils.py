######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
from numpy.linalg import norm
import os
from os.path import join,isfile,isdir
from transform_offsets import RotationTransformer,write_line_obj
from rotate_opt import rotate_opt
from pyquaternion import Quaternion
from vlz_example import VlzExample 
from body_angle_limits import clamp_angles, joint_limits
import PSpincalc as sp

BASESPLINE=9
MIDSPLINE=8
SHOULDERSPLINE=1

def qmul(q1,q2):
    q=np.zeros(4)
    q[0]=q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]
    q[1]=q1[0]*q2[1]+q1[1]*q2[0]+q1[2]*q2[3]-q1[3]*q2[2]
    q[2]=q1[0]*q2[2]-q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
    q[3]=q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
    return q

def frame_x(frame,x):
    return frame[:3,:3].dot(x)+frame[:3,3]

def to_array(q):
    return np.array([q[0],q[1],q[2],q[3]])

def from_axis_angle(axis,angle):
    q=np.array([1.,0.,0.,0.])
    # print('q',q)
    q[0]=np.cos(angle/2)
    q[1:]=axis*np.sin(angle/2)
    # print('from_axis_angle:',np.cos(angle/2),axis*np.sin(angle/2),q)
    return q

def rot_between_v(v1,v2):
    eps=1e-10
    R=np.array([1.,0.,0.,0.])
    v_outer=np.cross(v1,v2)
    v_outer_norm=norm(v_outer)

    v1_norm,v2_norm=norm(v1),norm(v2)
    if v_outer_norm<eps:
        return R
    if v1_norm<eps or v2_norm<eps:
        return R
    inner=np.inner(v1,v2)/(v1_norm*v2_norm)
    angle=np.arccos(inner)
    axis=v_outer/v_outer_norm
    q=from_axis_angle(axis,angle)
    return q

def qdis(q1,q2):
    q1=q1/norm(q1)
    q2=q2/norm(q2)
    inner=np.abs(np.inner(q1,q2))
    if inner>1:
        inner=1
    return np.arccos(inner)*2

def find_rot_angle(axis,v1,v2):
    # v1 -> v2
    v1=v1-np.inner(v1,axis)*axis
    v1/=norm(v1)
    v2=v2-np.inner(v2,axis)*axis
    v2/=norm(v2)
    cT=np.inner(v1,v2)
    t=np.arccos(cT)
    cross=np.cross(v1,v2)
    if cross[1]>=0:
        return t
    else:
        return -t

class JointSDKUtils:
    def __init__(self):
        self.n_joints=19
        # self.l=np.array([[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15]]).astype(int)
        self.l=np.array([[9,8],[8,1],[1,18],[18,0],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15]]).astype(int)
        # self.joint_map=['Head','Neck','LeftShoulder','LeftArm','LeftForeArm','RightShoulder','RightArm','RightForeArm','Spine','LowerBack','LHipjoint','LeftUpLeg','LeftLeg','RHipjoint','RightUpLeg','RightLeg']

        self.sdk_joint_parents=[-1 for i in range(self.n_joints)]
        for p,c in self.l:
            self.sdk_joint_parents[c]=p

        self.sdk_to_bld_joint_map=['Head','Spine','LeftShoulder','LeftArm','LeftForeArm','RightShoulder','RightArm','RightForeArm','LowerBack','Hips','LHipjoint','LeftUpLeg','LeftLeg','RHipjoint','RightUpLeg','RightLeg',None,None,'Neck']

        self.bld_joint_names=['Hips','LowerBack','Spine','Spine1','LeftShoulder','LeftArm','LeftForeArm','RightShoulder','RightArm','RightForeArm','Neck','Neck1','Head','RHipjoint','RightUpLeg','RightLeg','LHipjoint','LeftUpLeg','LeftLeg']
        self.bld_name_to_id={self.bld_joint_names[i]:i for i in range(len(self.bld_joint_names))}


    def load(self,path,max_frames=-1):
        frames=[]
        frame=[None for i in range(self.n_joints)]
        with open(path) as f:
            while True:
                line=f.readline()
                if line=='':
                    break
                parts=line.split(',')
                joint_id=int(parts[0])
                values=[float(v) for v in parts[1:]]
                p=np.array(values[:3])/100
                R=np.array(values[3:]).reshape((3,3))
                frame[joint_id]=(p,R)
                if joint_id==self.n_joints-1:
                    frames.append(frame)
                    frame=frame.copy()
                    if max_frames>=0 and len(frames)>=max_frames:
                        break
        return frames

    def write_bone_obj(self,obj_path,frame):
        print('write to',obj_path)
        with open(obj_path,'w') as f:
            for p,R in frame:
                f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
            for li in self.l:
                f.write('l {} {}\n'.format(li[0]+1,li[1]+1))

    def get_root_and_scale(self,frame,bone_structure):
        sdk_root=frame[BASESPLINE][0]
        bld_root=bone_structure.bone_dict['Hips'].world_frame[:3,3]
        sdk_scale=norm(frame[BASESPLINE][0]-frame[MIDSPLINE][0])+norm(frame[MIDSPLINE][0]-frame[SHOULDERSPLINE][0])
        bld_scale=norm(bone_structure.bone_dict['Hips'].local_tail)+norm(bone_structure.bone_dict['LowerBack'].local_tail)+norm(bone_structure.bone_dict['Spine'].local_tail)+norm(bone_structure.bone_dict['Spine1'].local_tail)
        return sdk_root,bld_root,bld_scale/sdk_scale

    def match_root_and_scale(self,frame,root_and_scale):
        sdk_root,bld_root,scale=root_and_scale
        new_frame=[]
        for p,R in frame:
            p=(p-sdk_root)*scale
            p[[0,2]]*=-1
            p+=bld_root
            new_frame.append((p,R))
        return new_frame

    def cvt_rotation(self,sdk_frame,bone_structure,root_and_scale):
        sdk_frame=self.match_root_and_scale(sdk_frame,root_and_scale)
        # self.write_bone_obj('joint_test/bone.obj',sdk_frame)
        result_rotations=[np.array([1.,0.,0.,0.]) for i in range(len(self.bld_joint_names))]
        
        bone_structure.reset()
        base_frame=bone_structure.bone_dict['Hips'].world_frame
        base_frame[:3,3]=sdk_frame[BASESPLINE][0]
        bone_structure.bone_dict['Hips'].set_world_frame(base_frame)
        bone_structure.apply_local_rotations({}) # update all frames

        local_frame_names=['LowerBack','Spine']
        local_frames=[bone_structure.bone_dict[name].local_frame for name in local_frame_names]
        src_world_pos=bone_structure.bone_dict['Neck'].world_frame[:,3]
        src_local_pos=np.linalg.inv(bone_structure.bone_dict['Spine'].world_frame).dot(src_world_pos)[:3]
        tgts=[(1,src_local_pos,sdk_frame[SHOULDERSPLINE][0])]
        R=rotate_opt(base_frame,local_frames,tgts)
        rotation_dict={local_frame_names[i]:Quaternion(R[i]).rotation_matrix for i in range(len(local_frame_names))}
        bone_structure.apply_local_rotations(rotation_dict)
        for i in range(len(R)):
            result_rotations[self.bld_name_to_id[local_frame_names[i]]]=R[i]

        for sdk_joint_id in [2,3,4,5,6,7,10,11,12,13,14,15]:
        # for sdk_joint_id in [2]:
            bone_name=self.sdk_to_bld_joint_map[sdk_joint_id]
            # print('sdk_joint_id',sdk_joint_id,'bone_name',bone_name)
            bone=bone_structure.bone_dict[bone_name]
            base_frame=bone.parent.world_frame

            local_frames=[bone.local_frame]
            src_local_pos=bone.local_tail
            tgt_world_pos=sdk_frame[sdk_joint_id][0]
            R=rotate_opt(base_frame,local_frames,[(0,src_local_pos,tgt_world_pos)])
            rotation_dict={bone_name:Quaternion(R[0]).rotation_matrix}
            bone_structure.apply_local_rotations(rotation_dict)
            result_rotations[self.bld_name_to_id[bone_name]]=R[0]

        # v,l=bone_structure.get_bone_obj()
        # obj_path='joint_test/cvt.obj'
        # print('write to',obj_path)
        # write_line_obj(obj_path,v,l)

        return np.array(result_rotations)

    def cvt_rotations(self,out_dir,sdk_frames,bone_structure):
        if not isdir(out_dir):
            os.makedirs(out_dir)
        first_frame=sdk_frames[0]
        root_and_scale=self.get_root_and_scale(first_frame,bone_structure)
        n_frames=len(sdk_frames)
        for frame_i in range(n_frames):
            sdk_frame=sdk_frames[frame_i]
            R=self.cvt_rotation(sdk_frame,bone_structure,root_and_scale)
            out_path=join(out_dir,'rotation_{:08d}.txt'.format(frame_i))
            print('write to',out_path)
            np.savetxt(out_path,R)

    def write_sdk_seq(self,out_dir,sdk_frames,bone_structure):
        if not isdir(out_dir):
            os.makedirs(out_dir)
        first_frame=sdk_frames[0]
        root_and_scale=self.get_root_and_scale(first_frame,bone_structure)
        for i in range(len(sdk_frames)):
            sdk_frame=sdk_frames[i]
            sdk_frame=self.match_root_and_scale(sdk_frame,root_and_scale)
            out_path=join(out_dir,'{:08d}.obj'.format(i))
            self.write_bone_obj(out_path,sdk_frame)

    def write_joint_seq(self,rot_dir,out_dir,bone_structure):
        if not isdir(out_dir):
            os.makedirs(out_dir)
        for frame_i in range(0,2248):
            rot_path=join(rot_dir,'rotation_{:08d}.txt'.format(frame_i))
            R=np.loadtxt(rot_path)
            rotation_dict={self.bld_joint_names[i]:Quaternion(R[i]).rotation_matrix for i in range(len(self.bld_joint_names))}
            bone_structure.reset()
            bone_structure.apply_local_rotations(rotation_dict)
            v,l=bone_structure.get_bone_obj()
            out_path=join(out_dir,'{:08d}.obj'.format(frame_i))
            print('write to',out_path)
            write_line_obj(out_path,v,l)

    def transfer_rotation(self,sdk_frame,bone_structure):
        root_bone=bone_structure.bone_dict['Hips']
        bone_structure.reset()
        sdk_joint_R=[None for i in range(self.n_joints)]
        rotations=[np.array([1.,0.,0.,0.]) for i in range(len(self.bld_joint_names))]

        # for debug
        # example=VlzExample('joint_test/debug')
        # line_width=8
        # axis_length=0.5

        bone_structure.reset()
        # for pi,ci in self.l:
        for pi,ci in [[9,8],[8,1],[1,18],[18,0],[1,2],[2,3],[1,5],[5,6],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15]]:
            if self.sdk_to_bld_joint_map[ci] is not None:
                bld_name=self.sdk_to_bld_joint_map[ci]
                bld_bone=bone_structure.bone_dict[bld_name]
                bld_id=self.bld_name_to_id[bld_name]

                bld_world_R=bld_bone.world_frame[:3,:3]
                world_v=sdk_frame[ci][0]-sdk_frame[pi][0]
                q=rot_between_v(bld_world_R[:,1],world_v)
                world_rot=Quaternion(q).rotation_matrix
                sdk_world_R=world_rot.dot(bld_world_R)
                sdk_joint_R[pi]=sdk_world_R
                if pi==BASESPLINE:
                    parent_world_R=root_bone.world_frame[:3,:3]
                else:
                    parent_world_R=sdk_joint_R[self.sdk_joint_parents[pi]]

                local_R=parent_world_R.T.dot(sdk_world_R)
                local_rot=bld_bone.local_frame[:3,:3].T.dot(local_R)
                rotations[bld_id]=Quaternion(matrix=local_rot).q
                bone_structure.apply_local_rotations({bld_name:local_rot})

                # vs=[p for p,R in sdk_frame]
                # colors=[np.array([1.,1.,1.]) for i in range(len(vs))]
                # example.draw_lines(vs,self.l,colors)

                # p=sdk_frame[pi][0]
                # example.proto_debug_utils.Add_Line(p,p+axis_length*sdk_world_R[:,0],color=[1.,0.,0.],width=line_width)
                # example.proto_debug_utils.Add_Line(p,p+axis_length*sdk_world_R[:,1],color=[0.,1.,0.],width=line_width)
                # example.proto_debug_utils.Add_Line(p,p+axis_length*sdk_world_R[:,2],color=[0.,0.,1.],width=line_width)
                # v,l=bone_structure.get_bone_obj()
                # example.draw_lines(v+np.array([7.,0,0]),l,line_colors=np.ones(v.shape))
                # example.add_frame()

        for gpi,pi,ci in [[2,3,4],[5,6,7]]:
            bld_name=self.sdk_to_bld_joint_map[ci]
            bld_bone=bone_structure.bone_dict[bld_name]
            bld_id=self.bld_name_to_id[bld_name]

            vpc=sdk_frame[ci][0]-sdk_frame[pi][0]
            vgpp=sdk_frame[gpi][0]-sdk_frame[pi][0]
            cT=-np.inner(vpc,vgpp)/norm(vpc)/norm(vgpp)
            t=np.arccos(cT)
            qc=from_axis_angle(np.array([1,0,0]),t)
            rotations[bld_id]=qc
            Rc=Quaternion(qc).rotation_matrix
            bone_structure.apply_local_rotations({bld_name:Rc})

            # vs=[p for p,R in sdk_frame]
            # colors=[np.array([1.,1.,1.]) for i in range(len(vs))]
            # example.draw_lines(vs,self.l,colors)
            # v,l=bone_structure.get_bone_obj()
            # colors=np.ones((len(v),3))
            # example.draw_lines(v+np.array([8,0,0]),l,colors)
            # example.add_frame()

            bld_name_p=self.sdk_to_bld_joint_map[pi]
            bld_bone_p=bone_structure.bone_dict[bld_name_p]
            bld_id_p=self.bld_name_to_id[bld_name_p]
            vpc_src=Rc[:,1]
            vpc_tgt=bld_bone_p.world_frame[:3,:3].T.dot(vpc)
            t=find_rot_angle(np.array([0,1,0]),vpc_src,vpc_tgt)
            qp=qmul(rotations[bld_id_p],from_axis_angle(np.array([0,1,0]),t))
            Rp=Quaternion(qp).rotation_matrix
            bone_structure.apply_local_rotations({bld_name_p:Rp})
            rotations[bld_id_p]=qp

            # colors=[np.array([1.,1.,1.]) for i in range(len(vs))]
            # example.draw_lines(vs,self.l,colors)
            # v,l=bone_structure.get_bone_obj()
            # colors=np.ones((len(v),3))
            # example.draw_lines(v+np.array([8,0,0]),l,colors)
            # example.add_frame()

        return np.array(rotations)

    def transfer_rotations(self,out_dir,sdk_frames,bone_structure):
        if not isdir(out_dir):
            os.makedirs(out_dir)
        first_frame=sdk_frames[0]
        root_and_scale=self.get_root_and_scale(first_frame,bone_structure)
        n_frames=len(sdk_frames)
        for frame_i in range(n_frames):
            sdk_frame=sdk_frames[frame_i]
            sdk_frame=self.match_root_and_scale(sdk_frame,root_and_scale)
            rotations=self.transfer_rotation(sdk_frame,bone_structure)
            out_path=join(out_dir,'rotation_{:08d}.txt'.format(frame_i))
            print('save to',out_path)
            np.savetxt(out_path,rotations)

    def clamp_rotation(self,frames,raw_rot_dir,clamp_rot_dir):
        if not isdir(clamp_rot_dir):
            os.makedirs(clamp_rot_dir)
        for frame_i in frames:
            raw_rot_path=join(raw_rot_dir,'rotation_{:08d}.txt'.format(frame_i))
            rot=np.loadtxt(raw_rot_path)
            for bone_i in range(len(self.bld_joint_names)):
                bone_name=self.bld_joint_names[bone_i]

                # hack
                # if bone_name=='LeftForeArm' or bone_name=='RightForeArm':
                #     continue

                rot[bone_i]=clamp_angles(rot[bone_i],bone_name)
            clamp_rot_path=join(clamp_rot_dir,'rotation_{:08d}.txt'.format(frame_i))
            print('save to',clamp_rot_path)
            np.savetxt(clamp_rot_path,rot)

    def draw_seq(self,out_dir,sdk_frames,raw_rot_dir,clamp_rot_dir,bone_structure):
        example=VlzExample(out_dir)
        first_frame=sdk_frames[0]
        root_and_scale=self.get_root_and_scale(first_frame,bone_structure)
        n_frames=len(sdk_frames)
        for frame_i in range(n_frames):
            sdk_frame=sdk_frames[frame_i]
            sdk_frame=self.match_root_and_scale(sdk_frame,root_and_scale)
            vs=[p for p,R in sdk_frame]
            colors=[np.array([1.,1.,1.]) for i in range(len(vs))]
            example.draw_lines(vs,self.l,colors)

            raw_rot_path=join(raw_rot_dir,'rotation_{:08d}.txt'.format(frame_i))
            raw_rot=np.loadtxt(raw_rot_path)
            clamp_rot_path=join(clamp_rot_dir,'rotation_{:08d}.txt'.format(frame_i))
            clamp_rot=np.loadtxt(clamp_rot_path)
            rot_dis=[qdis(raw_rot[i],clamp_rot[i]) for i in range(len(clamp_rot))]
            rot_dis=np.array(rot_dis)

            rotation_dict={self.bld_joint_names[i]:Quaternion(raw_rot[i]).rotation_matrix for i in range(len(self.bld_joint_names))}
            bone_structure.reset()
            bone_structure.apply_local_rotations(rotation_dict)
            v,l=bone_structure.get_bone_obj()
            t=np.clip(rot_dis/0.15,0,1).reshape(-1,1)
            colors0=np.tile(np.array([1.,1.,1.]),(len(l),1))
            colors1=np.tile(np.array([1.,0.,0.]),(len(l),1))
            colors=colors0*(1-t)+colors1*t
            example.draw_lines(v+np.array([8,0,0]),l,colors)

            rotation_dict={self.bld_joint_names[i]:Quaternion(clamp_rot[i]).rotation_matrix for i in range(len(self.bld_joint_names))}
            bone_structure.reset()
            bone_structure.apply_local_rotations(rotation_dict)
            v,l=bone_structure.get_bone_obj()
            colors=[np.array([1.,1.,1.]) for i in range(len(l))]
            example.draw_lines(v+np.array([16,0,0]),l,colors)

            example.add_frame()


if __name__=='__main__':
    sdk_utils=JointSDKUtils()
    rt=RotationTransformer()

    frames=sdk_utils.load('joint_test/orbbec_joint_record_2.csv')
    # sdk_utils.write_bone_obj('joint_test/bone.obj',frames[0])
    # sdk_utils.cvt_rotation(frames[0],rt.bone_structure)
    # sdk_utils.cvt_rotations('joint_test/bld_rots',frames,rt.bone_structure)
    # sdk_utils.write_skeleton_sequence('joint_test/sdk_seq',frames,rt.bone_structure)
    # sdk_utils.write_joint_seq('joint_test/bld_rots','joint_test/bld_seq',rt.bone_structure)

    start,end=0,2248
    sdk_utils.transfer_rotations('joint_test/raw_rot',frames[start:end],rt.bone_structure)
    sdk_utils.clamp_rotation(range(0,end-start),'joint_test/raw_rot','joint_test/clamp_rot')
    sdk_utils.draw_seq('joint_test/joint_dbg',frames[start:end],'joint_test/raw_rot','joint_test/clamp_rot',rt.bone_structure)
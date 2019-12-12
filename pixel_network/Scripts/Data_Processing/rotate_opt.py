######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import numpy as np
from pyquaternion import Quaternion
from transform_offsets import write_line_obj

def qmul(q1,q2):
    q=torch.zeros(4).to(dtype=q1.dtype)
    q[0]=q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]
    q[1]=q1[0]*q2[1]+q1[1]*q2[0]+q1[2]*q2[3]-q1[3]*q2[2]
    q[2]=q1[0]*q2[2]-q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
    q[3]=q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
    return q

def conjugate(q):
    q=q.clone()
    q[1:]*=-1
    return q

def rotate_v(q,v):
    vq=torch.zeros(4).to(dtype=q.dtype)
    vq[1:]=v
    return qmul(q,qmul(vq,conjugate(q)))[1:]

def frame_v(frame,v):
    R,t=frame
    return rotate_v(R,v)+t

def from_axis_angle(axis,angle):
    q=torch.zeros(4).to(dtype=axis.dtype)
    q[0]=np.cos(angle/2)
    q[1:]=np.sin(angle/2)*axis
    return q

def rotate_opt_single_joint(base_frame,local_frame,tgt):
    eps=1e-8
    _,local_src,world_tgt=tgt
    local_src=torch.from_numpy(local_src)
    world_tgt=torch.from_numpy(world_tgt)
    base_R,base_t=base_frame
    local_R,local_t=local_frame
    world_R=qmul(base_R,local_R)
    world_t=frame_v(base_frame,local_t)
    local_tgt=rotate_v(conjugate(world_R),world_tgt-world_t)

    world_tgt=frame_v((world_R,world_t),local_tgt)
    world_src=frame_v((world_R,world_t),local_src)
    write_line_obj('joint_test/test.obj',np.array([world_tgt.numpy(),world_src.numpy()]),np.array([[0,1]]).astype(int))

    axis=torch.cross(local_src,local_tgt)
    sin_t=torch.norm(axis)
    R=torch.tensor([1,0,0,0],dtype=axis.dtype)
    if sin_t<eps:
        return R
    axis/=sin_t
    norm_src=torch.norm(local_src)
    norm_tgt=torch.norm(local_tgt)
    if norm_src<eps or norm_tgt<eps:
        return R
    cos_t=torch.sum(local_src*local_tgt)/(norm_src*norm_tgt)
    theta=torch.acos(cos_t)
    return from_axis_angle(axis,theta).numpy()


def rotate_opt(base_frame,local_frames,tgts,iters=35,lr=1e-2):
    def get_frame(m):
        # print(np.linalg.norm(np.linalg.inv(m[:3,:3])-m[:3,:3].T))
        q=Quaternion(matrix=m[:3,:3])
        return torch.from_numpy(np.array([q[0],q[1],q[2],q[3]])),torch.from_numpy(m[:3,3])
    base_frame=get_frame(base_frame)
    local_frames=[get_frame(frame) for frame in local_frames]

    if len(local_frames)==1:
        return [rotate_opt_single_joint(base_frame,local_frames[0],tgts[0])]

    R=torch.zeros(len(local_frames),4).to(dtype=base_frame[0].dtype)
    R[:,0]=1
    R.requires_grad_(True)
    for opt_iter in range(iters):
        world_frames=[]
        for i in range(len(local_frames)):
            local_R,local_t=local_frames[i]
            local_R=qmul(local_R,R[i])
            if i==0:
                prev_R,prev_t=base_frame
            else:
                prev_R,prev_t=world_frames[i-1]
            world_R=qmul(prev_R,local_R)
            world_t=frame_v((prev_R,prev_t),local_t)
            world_frames.append((world_R,world_t))

        loss=0
        for frame_i,local_pos,tgt_pos in tgts:
            local_pos=torch.from_numpy(local_pos)
            tgt_pos=torch.from_numpy(tgt_pos)
            world_frame=world_frames[frame_i]
            cur_pos=frame_v(world_frame,local_pos)
            loss+=torch.sum((cur_pos-tgt_pos)**2)
        # print('iter:',opt_iter,'loss',loss.item())
        R.grad=None
        loss.backward()
        R.data-=R.grad.data*lr
        R.data/=torch.norm(R,dim=1,keepdim=True).data
    print('loss',loss.item())
    return R.detach().numpy()
######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
import PSpincalc as sp

angle_limits = {
    "NECK_ROTATION": 20,
    "NECK_FLEXION": 34/2,
    "NECK_EXTENSION": 40/2,
    "NECK_LATERAL_BEND": 30/2,

    "SHOULDER_ABDUCTION": 20,
    "SHOULDER_ADDUCTION": 80,

    "SHOULDER_ROTATION_LATERAL": 40/2,
    "SHOULDER_ROTATION_MEDIAL": 20/2, 
     
    "SHOULDER_FLEXION": 90,
    "SHOULDER_EXTENSION": 130,

    "ELBOW_FLEXION": 140/2,
    "FOREARM_PRONATION": 50/4,
    "FOREARM_SUPINATION": 80/4,

    "SPINE_FLEXION": 75/2,
    "SPINE_EXTENSION": 30/2,
    "SPINE_LATERAL_BENDING": 35/2,
    "SPINE_TWIST": 45/2
    }

angle_stds={
    'LowerBackX':[angle_limits['SPINE_FLEXION']/2,angle_limits['SPINE_EXTENSION']/2],
    'LowerBackZ':[angle_limits['SPINE_LATERAL_BENDING']/2,angle_limits['SPINE_LATERAL_BENDING']/2],
    'LowerBackY':[angle_limits['SPINE_TWIST']/2,angle_limits['SPINE_TWIST']/2],
    'SpineX':[angle_limits['SPINE_FLEXION']/2,angle_limits['SPINE_EXTENSION']/2],
    'SpineZ':[angle_limits['SPINE_LATERAL_BENDING']/2,angle_limits['SPINE_LATERAL_BENDING']/2],
    'SpineY':[angle_limits['SPINE_TWIST']/2,angle_limits['SPINE_TWIST']/2],
    'NeckX':[angle_limits['NECK_FLEXION']/2,angle_limits['NECK_EXTENSION']/4],
    'Neck1Y':[angle_limits['NECK_ROTATION'],angle_limits['NECK_ROTATION']],
    'Neck1Z':[angle_limits['NECK_LATERAL_BEND'],angle_limits['NECK_LATERAL_BEND']],
    'Neck1X':[angle_limits['NECK_FLEXION']/2,angle_limits['NECK_EXTENSION']/4*3],
    'LeftShoulderX':[angle_limits['SHOULDER_ADDUCTION']/5,angle_limits['SHOULDER_ABDUCTION']/5],
    'LeftShoulderZ':[angle_limits['SHOULDER_FLEXION']/4,angle_limits['SHOULDER_EXTENSION']/4],
    'LeftShoulderY':[angle_limits['SHOULDER_ROTATION_MEDIAL']/4,angle_limits['SHOULDER_ROTATION_LATERAL']/8],
    'LeftArmX':[angle_limits['SHOULDER_ADDUCTION']/5*4,angle_limits['SHOULDER_ABDUCTION']],
    'LeftArmZ':[angle_limits['SHOULDER_FLEXION']/4*3,angle_limits['SHOULDER_EXTENSION']],
    'LeftArmY':[angle_limits['SHOULDER_ROTATION_LATERAL']/4*3,angle_limits['SHOULDER_ABDUCTION']/8*7],
    'LeftForeArmX':[angle_limits['ELBOW_FLEXION'],0],
    'LeftForeArmY':[angle_limits['FOREARM_PRONATION'],angle_limits['FOREARM_SUPINATION']],
    'RightShoulderX':[angle_limits['SHOULDER_ADDUCTION']/5,angle_limits['SHOULDER_ABDUCTION']/5],
    'RightShoulderZ':[angle_limits['SHOULDER_FLEXION']/4,angle_limits['SHOULDER_EXTENSION']/4],
    'RightShoulderY':[angle_limits['SHOULDER_ROTATION_LATERAL']/8,angle_limits['SHOULDER_ROTATION_MEDIAL']/4],
    'RightArmX':[angle_limits['SHOULDER_ADDUCTION']/5*4,angle_limits['SHOULDER_ABDUCTION']/5*4],
    'RightArmZ':[angle_limits['SHOULDER_FLEXION']/4*3,angle_limits['SHOULDER_EXTENSION']/4*3],
    'RightArmY':[angle_limits['SHOULDER_ROTATION_LATERAL']/8*7,angle_limits['SHOULDER_ROTATION_LATERAL']/8*7],
    'RightForeArmX':[angle_limits['ELBOW_FLEXION'],0],
    'RightForeArmY':[angle_limits['FOREARM_SUPINATION'],angle_limits['FOREARM_PRONATION']]
}

for k,v in angle_stds.items():
    angle_stds[k]=np.array([-v[1],v[0]])/180*np.pi

joint_limits=angle_stds
# joint_limits['LeftArmY']=np.array([-np.pi/2,np.pi/2])
# joint_limits['LeftArmZ'][1]=np.pi/2
# joint_limits['RightArmY']=np.array([-np.pi/2,np.pi/2])
# joint_limits['RightArmZ'][1]=np.pi/2
# joint_limits['LeftForeArmX'][1]=140/180*np.pi
# joint_limits['RightForeArmX'][1]=140/180*np.pi

joint_offsets={
    'LeftShoulderX':-15/180*np.pi,
    'RightShoulderX':20/180*np.pi,
    'LeftShoulderZ':20/180*np.pi,
    'RightShoulderZ':-15/180*np.pi,
    'Spine1':10/180*np.pi
}

angle_modes={
    'LowerBack':'yxz',
    'Spine':'yzx',
    'Neck':'zyx',
    'Neck1':'yxz',
    'LeftShoulder':'xzy',
    'LeftArm':'xzy',
    'LeftForeArm':'zxy',
    'RightShoulder':'xzy',
    'RightArm':'xzy',
    'RightForeArm':'zxy'
}

# for gaussians, x2
for name in ['NeckX','Neck1Y','Neck1Z','Neck1X','LeftShoulderY','LeftArmY','RightShoulderY','RightArmY','LowerBackX','LowerBackY','LowerBackZ','SpineX','SpineY','SpineZ']:
    joint_limits[name]*=2

def clamp_angles(q,bone_name):
    if not bone_name in angle_modes:
        return q
    try:
        mode=angle_modes[bone_name]
        angles=sp.Q2EA(q,mode)[0]
        for i in range(3):
            m=mode[i]
            angle=angles[i]
            name='{}{}'.format(bone_name,m.upper())
            if name in joint_offsets:
                angle+=joint_offsets[name]

            if name in joint_limits:
                l,u=joint_limits[name]
                if angle<l:
                    angle=l
                if angle>u:
                    angle=u
            else:
                angle=0
            angles[i]=angle
        return sp.EA2Q(angles,mode)
    except:
        print('return input')
        return q

# def qmult(q1, q2):
#     w1, x1, y1, z1 = q1
#     w2, x2, y2, z2 = q2
#     w = w1*w2 - x1*x2 - y1*y2 - z1*z2
#     x = w1*x2 + x1*w2 + y1*z2 - z1*y2
#     y = w1*y2 + y1*w2 + z1*x2 - x1*z2
#     z = w1*z2 + z1*w2 + x1*y2 - y1*x2
#     return [w, x, y, z]

# def angle2quat(x,y,z,a):
#     t = np.deg2rad(a)    
#     return [np.cos(t/2), x*np.sin(t/2), y*np.sin(t/2), z*np.sin(t/2)]

# def test():
#     q=qmult(qmult(angle2quat(0,1,0,10),angle2quat(1,0,0,20)),angle2quat(0,0,1,30))
#     print('yxz',sp.Q2EA(q,'yxz')/3.14*180)
#     # print('xyz',sp.Q2EA(q,'xyz')/3.14*180)
#     # print('zyx',sp.Q2EA(q,'zyx')/3.14*180)

# if __name__=='__main__':
#     test()
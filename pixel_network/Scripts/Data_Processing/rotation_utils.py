######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import numpy as np
import pyquaternion

def read_quaternions_from_file(filename, num_joints=10):
    if not os.path.exists(filename):
        print('%s is missing' %filename)
        return None
    
    quaternions = np.zeros([num_joints, 4])
    with open(filename, 'r') as fin:
        index = 0
        for line in fin:
            q = line.strip().split()
            assert(len(q) == 4)
            for j in range(4):
                quaternions[index, j] = float(q[j])
            index += 1
    if index != num_joints:
        print('wrong number of joints %d not equal to %d' %(index, num_joints))
    return quaternions


def quaternion_to_rotation_matrix(q):
    assert(q.size == 4)
    norm = np.linalg.norm(q)
    assert(np.abs(norm - 1) < 1e-4)
    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]
    
#     R = np.zeros([3, 3])
#     R[0, 0] = a*a + b*b - c*c - d*d
#     R[0, 1] = 2*b*c - 2*a*d
#     R[0, 2] = 2*b*d + 2*a*c
#     R[1, 0] = 2*b*c + 2*a*d
#     R[1, 1] = a*a - b*b + c*c - d*d
#     R[1, 2] = 2*c*d - 2*a*b
#     R[2, 0] = 2*b*d - 2*a*c
#     R[2, 1] = 2*c*d + 2*a*b
#     R[2, 2] = a*a - b*b - c*c + d*d
    
    quat = pyquaternion.Quaternion(a, b, c, d)
    R1 = quat.rotation_matrix
    
#     print(R)
#     print(R1)
#     print(np.linalg.norm(R-R1, ord='fro'))
    
    return R1
######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
import os

class Obj:
    # 0-based indexing
    def __init__(self, v=None, f=None, vt=None):
        self.v = v
        self.f = f
        self.vt = vt

def read_obj(obj_filename):
    if not os.path.exists(obj_filename):
        return None
    verts = []
    vtexs = []
    faces = []

    with open(obj_filename) as fin:
        for line in fin:
            if line.startswith('v '):
                fields = line.strip().split()
                assert(len(fields) == 4)
                verts.append([float(fields[1]), float(fields[2]), float(fields[3])])
            if line.startswith('vt '):
                fields=line.strip().split()
                vtexs.append([float(fields[1]), float(fields[2])])
            if line.startswith('f '):
                fields = line.strip().split()
                assert(len(fields) == 4)
                id_strs=[fields[i].split('/')[0] for i in range(1,4)]
                faces.append([int(id_strs[0])-1, int(id_strs[1])-1, int(id_strs[2])-1])
        verts = np.array(verts)
        vtexs = np.array(vtexs)
        faces = np.array(faces)
        nv = len(verts)
#         assert(np.amax(faces) == nv - 1)
#         assert(np.amin(faces) == 0)
    return Obj(verts, faces, vtexs)

def write_obj(obj, obj_filename):
    assert(obj.v is not None)
    with open(obj_filename, 'w') as fout:
        for i in range(obj.v.shape[0]):
            fout.write('v %f %f %f\n' %(obj.v[i][0], obj.v[i][1], obj.v[i][2]))
        if obj.vt is None or len(obj.vt)==0:
            if obj.f is not None:
                for i in range(obj.f.shape[0]):
                    fout.write('f %d %d %d\n' %(obj.f[i][0]+1, obj.f[i][1]+1, obj.f[i][2]+1))
        else:
            for vt in obj.vt:
                fout.write('vt {} {}\n'.format(vt[0],vt[1]))
            if obj.f is not None:
                for fi in obj.f:
                    fout.write('f {0}/{0} {1}/{1} {2}/{2}\n'.format(fi[0]+1,fi[1]+1,fi[2]+1))

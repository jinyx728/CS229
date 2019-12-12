######################################################################
# Copyright 2017. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np

def write_ply(filename, vertices, faces, colors=None):
    with open(filename, 'w') as f:
        num_vertices = len(vertices)
        num_faces = len(faces)
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(num_vertices))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if colors is not None:
            assert(len(colors)==len(vertices))
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('element face {}\n'.format(num_faces))
        f.write('property list uchar int vertex_index\n')
        f.write('end_header\n')
        if colors is not None:
            for i in range(num_vertices):
                v = vertices[i]
                c = colors[i]
                f.write('{} {} {} {} {} {}\n'.format(v[0], v[1], v[2], c[0], c[1], c[2]))
        else:
            for i in range(num_vertices):
                v = vertices[i]
                f.write('{} {} {}\n'.format(v[0], v[1], v[2]))
        for face in faces:
            n = len(face)
            f.write('{} '.format(n))
            for i in face:
                f.write('{} '.format(i))
            f.write('\n')
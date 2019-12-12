######################################################################
# Copyright 2019. Yongxu Jin.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import argparse
import math
import numpy as np
import trimesh
import os

def subdivide(objfile, outpath, front_file, back_file, bdry_file):
    mesh = trimesh.load(objfile, process=False)
    vertices = mesh.vertices
    faces = mesh.faces
    v_list = []
    f_list = []
    v_index = np.full(len(vertices), -1)  # old index -> new index
    v_index_new = np.full((len(vertices), len(vertices)), -1)

    front_vertices = np.loadtxt(front_file)
    back_vertices = np.loadtxt(back_file)
    bdry_vertices = np.loadtxt(bdry_file)

    # new front, back and bdry txt
    front_list = []
    back_list = []
    bdry_list = []

    i = 0
    for face in faces:
        # store vertex    
        for j in range(3):
            idx = face[j]
            if v_index[idx] == -1:
                v_index[idx] = i
                v_list.append("v {} {} {}\n".format(vertices[idx][0], vertices[idx][1], vertices[idx][2]))
                # store txt
                if np.isin(idx, front_vertices):
                    front_list.append(str(i)+"\n")
                if np.isin(idx, back_vertices):
                    back_list.append(str(i)+"\n")
                if np.isin(idx, bdry_vertices):
                    bdry_list.append(str(i)+"\n")
                i += 1 
        
        # new vertex
        for j in range(3):
            idx1 = face[j%3]
            idx2 = face[(j+1)%3]
            if v_index_new[idx1, idx2] == -1:
                v_index_new[idx1, idx2] = i
                v_index_new[idx2, idx1] = i
                v_new = (vertices[idx1] + vertices[idx2]) / 2
                v_list.append("v {} {} {}\n".format(v_new[0], v_new[1], v_new[2]))
                # store txt
                if np.isin(idx1, front_vertices) and np.isin(idx2, front_vertices):
                    front_list.append(str(i)+"\n")
                if np.isin(idx1, back_vertices) and np.isin(idx2, back_vertices):
                    back_list.append(str(i)+"\n")
                if np.isin(idx1, bdry_vertices) and np.isin(idx2, bdry_vertices):
                    bdry_list.append(str(i)+"\n")
                i += 1

        # new faces (index from 1)
        i0 = v_index[face[0]] + 1
        i1 = v_index[face[1]] + 1
        i2 = v_index[face[2]] + 1
        i01 = v_index_new[face[0], face[1]] + 1
        i02 = v_index_new[face[0], face[2]] + 1
        i12 = v_index_new[face[1], face[2]] + 1
        f_list.append("f {} {} {}\n".format(i0, i01, i02))
        f_list.append("f {} {} {}\n".format(i01, i1, i12))
        f_list.append("f {} {} {}\n".format(i01, i12,i02))
        f_list.append("f {} {} {}\n".format(i02, i12, i2))

    # write file
    outf = open(os.path.join(outpath, "flat_tshirt.obj"), "w")
    outf.writelines(v_list)
    outf.write("\n")
    outf.writelines(f_list)
    outf.close()

    frontf = open(os.path.join(outpath, "front_vertices.txt"), "w")
    frontf.writelines(front_list)
    frontf.close()

    backf = open(os.path.join(outpath, "back_vertices.txt"), "w")
    backf.writelines(back_list)
    backf.close()

    bdryf = open(os.path.join(outpath, "bdry_vertices.txt"), "w")
    bdryf.writelines(bdry_list)
    bdryf.close()
    return

if __name__ == "__main__":
    input_path = '/phoenix/yxjin/shared_data/flat_tshirt.obj'
    front_vertices = '/phoenix/yxjin/shared_data/front_vertices.txt'
    back_vertices = '/phoenix/yxjin/shared_data/back_vertices.txt'
    bdry_vertices = '/phoenix/yxjin/shared_data/bdry_vertices.txt'
    output_path = '/phoenix/yxjin/shared_data_highres'
    subdivide(input_path, output_path, front_vertices, back_vertices, bdry_vertices)
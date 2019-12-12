######################################################################
# Copyright 2019. Yongxu Jin.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import argparse
import math
import numpy as np
import trimesh

def subdivide(objfile, outfile, texfile):
    mesh = trimesh.load(objfile, process=False)
    vertices = mesh.vertices
    faces = mesh.faces
    v_list = []
    vt_list = []
    f_list = []
    v_index = np.full(len(vertices), -1)  # old index -> new index
    v_index_new = np.full((len(vertices), len(vertices)), -1)

    # load default texture coordinate
    vtf = open(texfile, "r")
    tex_coord = vtf.readlines()
    vtf.close()
    for i in range(len(tex_coord)):
        split = tex_coord[i].split(" ")
        tex_coord[i] = np.array([float(split[1]), float(split[2])])

    i = 0
    for face in faces:
        # store vertex    
        for j in range(3):
            idx = face[j]
            if v_index[idx] == -1:
                v_index[idx] = i
                v_list.append("v {} {} {}\n".format(vertices[idx][0], vertices[idx][1], vertices[idx][2]))
                vt_list.append("vt {} {}\n".format(tex_coord[idx][0], tex_coord[idx][1]))
                i += 1 
        
        # new vertex
        for j in range(3):
            idx1 = face[j%3]
            idx2 = face[(j+1)%3]
            if v_index_new[idx1, idx2] == -1:
                v_index_new[idx1, idx2] = i
                v_index_new[idx2, idx1] = i
                v_new = (vertices[idx1] + vertices[idx2]) / 2
                vt_new = (tex_coord[idx1] + tex_coord[idx2]) / 2
                v_list.append("v {} {} {}\n".format(v_new[0], v_new[1], v_new[2]))
                vt_list.append("vt {} {}\n".format(vt_new[0], vt_new[1]))
                i += 1

        # new faces (index from 1)
        i0 = v_index[face[0]] + 1
        i1 = v_index[face[1]] + 1
        i2 = v_index[face[2]] + 1
        i01 = v_index_new[face[0], face[1]] + 1
        i02 = v_index_new[face[0], face[2]] + 1
        i12 = v_index_new[face[1], face[2]] + 1
        f_list.append("f {}/{} {}/{} {}/{}\n".format(i0, i0, i01, i01, i02, i02))
        f_list.append("f {}/{} {}/{} {}/{}\n".format(i01, i01, i1, i1, i12, i12))
        f_list.append("f {}/{} {}/{} {}/{}\n".format(i01, i01, i12, i12, i02, i02))
        f_list.append("f {}/{} {}/{} {}/{}\n".format(i02, i02, i12, i12, i2, i2))
    
    # write into an obj file
    outf = open(outfile, "w")
    outf.write("mtllib tshirt.mtl\n")
    outf.writelines(v_list)
    outf.write("\n")
    outf.writelines(vt_list)
    outf.write("\nusemtl material0\n")
    outf.writelines(f_list)
    outf.close()
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input obj file", required=True)
    parser.add_argument("-o", "--output", help="output obj file", required=True)
    parser.add_argument("-t", "--texture", help="default texture coordinate", required=True)
    args = parser.parse_args()
    subdivide(args.input, args.output, args.texture)
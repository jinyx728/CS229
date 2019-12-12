######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import numpy as np
import cv2
from render_offsets import get_vts_normalize_m, normalize_vts

def draw_edges_on_img(img,vts,edges,color,thickness):
    H=img.shape[0]

    def vts_to_pix_pos(vts):
        vts=vts.copy()
        vts[:,1]*=-1
        pos=np.around((vts+1)/2*H-0.5).astype(int)
        pix_pos=[]
        for p in pos:
            pix_pos.append((p[0],p[1]))
        return pix_pos

    pix_pos=vts_to_pix_pos(vts)
    for edge in edges:
        i0,i1=edge[0],edge[1]
        p0,p1=pix_pos[i0],pix_pos[i1]
        cv2.line(img,p0,p1,color=color,thickness=thickness)

def read_obj(obj_path):
    if not os.path.exists(obj_path):
        assert(False)
    verts = []
    edges = []
    with open(obj_path) as fin:
        for line in fin:
            if line.startswith('v '):
                fields = line.strip().split()
                assert(len(fields) == 4)
                verts.append([float(fields[1]), float(fields[2]), float(fields[3])])
            if line.startswith('l '):
                fields = line.strip().split()
                edges.append((int(fields[1])-1, int(fields[2])-1))
        verts = np.array(verts)
    return verts,edges

def draw_edges(img_path,obj_path,out_path,color=(0,0,0),thickness=2):
    img=cv2.imread(img_path)
    vts,edges=read_obj(obj_path)
    normalize_m=get_vts_normalize_m(vts)
    vts=normalize_vts(vts[:,:2],normalize_m)

    draw_edges_on_img(img,vts,edges,color=color,thickness=thickness)
    vts[:,0]+=2
    draw_edges_on_img(img,vts,edges,color=color,thickness=thickness)
    vts[:,0]+=2
    draw_edges_on_img(img,vts,edges,color=color,thickness=thickness)

    cv2.imwrite(out_path,img)

def split_img(in_path,out_paths):
    img=cv2.imread(in_path)
    total_W=img.shape[1]
    W=total_W//len(out_paths)
    for i in range(len(out_paths)):
        out_path=out_paths[i]
        cv2.imwrite(out_path,img[:,W*i:W*(i+1),:])

if __name__=='__main__':
    # split_img('plots/whole_w_pad.png',['plots/front_w_pad.png','plots/back_w_pad.png'])
    draw_edges('plots/front_w_pad.png','plots/front_bdry_edges.obj','plots/front_w_bdry.png')
    draw_edges('plots/back_w_pad.png','plots/back_bdry_edges.obj','plots/back_w_bdry.png')




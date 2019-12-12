######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
from obj_io import Obj, write_obj

def read_obj(file_path):
    vts=[]
    fcs=[]
    with open(file_path) as f:
        while True:
            line=f.readline()
            #print('line', line)
            if line == '':
                break
            parts=line.split()
            if len(parts) == 0:
                continue
            if parts[0]=='v':
                vts.append([float(parts[1]),float(parts[2]),float(parts[3])])
            elif parts[0]=='f':
                fc=[]
                for i in range(1,len(parts)):
                    fc.append(int(parts[i])-1)
                fcs.append(fc)
    return np.array(vts),fcs

def split_fcs(vts,fcs):
    print('len(vts)',len(vts))
    splitted_fcs=[]
    for fc in fcs:
        if len(fc)==3:
            splitted_fcs.append(fc)
            continue
        elif len(fc)!=4:
            print('some thing is wrong with fc',fc)
            continue
        # sums=[vts[vt_id][0]+vts[vt_id][1] for vt_id in fc]
        # min_id=np.argmin(np.array(sums))
        # vt_ids=[(min_id+i)%len(fc) for i in range(len(fc))]
        # splitted_fcs.append(vt_ids[0],vt_ids[1],vt_ids[2])
        # splitted_fcs.append(vt_ids[0],vt_ids[2],vt_ids[3])
        v1=vts[fc[2]]-vts[fc[0]]
        v2=vts[fc[3]]-vts[fc[1]]
        s1=np.abs(v1[0]-v1[1])/np.linalg.norm(np.array([v1[0],v1[1]]))
        s2=np.abs(v2[0]-v2[1])/np.linalg.norm(np.array([v2[0],v2[1]]))
        if s1>s2:
            splitted_fcs.append([fc[0],fc[1],fc[2]])
            splitted_fcs.append([fc[0],fc[2],fc[3]])
        else:
            splitted_fcs.append([fc[0],fc[1],fc[3]])
            splitted_fcs.append([fc[1],fc[2],fc[3]])
    return np.array(splitted_fcs)

if __name__=='__main__':
    # vts,fcs=read_obj('../../shared_data/front_flat_TshirtW_remesh3_lowres_quad.obj')
    # vts,fcs=read_obj('../../shared_data/back_flat_TshirtW_remesh3_lowres_quad.obj')
    # vts,fcs=read_obj('../../../project_data/midres_tshirt/front_flat_midres_quad.obj')
    vts,fcs=read_obj('../../../project_data/midres_tshirt/back_flat_midres_quad.obj')
    fcs=split_fcs(vts,fcs)
    obj=Obj(v=vts,f=fcs)
    # write_obj(obj,'../../shared_data/front_flat_TshirtW_remesh3_lowres_tri.obj')
    # write_obj(obj,'../../shared_data/back_flat_TshirtW_remesh3_lowres_tri.obj')
    # write_obj(obj,'../../../project_data/midres_tshirt/front_flat_midres_tri.obj')
    write_obj(obj,'../../../project_data/midres_tshirt/back_flat_midres_tri.obj')


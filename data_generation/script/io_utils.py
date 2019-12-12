import numpy as np

def read_vt(in_path):
    vt=[]
    with open(in_path) as f:
        lines=f.readlines()
        for line in lines:
            parts=line.split()
            vt.append([float(parts[1]),float(parts[2])])
    return np.array(vt)

def write_vt(out_path,vt):
    with open(out_path,'w') as f:
        for vti in vt:
            f.write('vt {} {}\n'.format(vti[0],vti[1]))

def write_uv(out_path,vt,fcs):
    with open(out_path,'w') as f:
        for i in fcs.reshape(-1):
            f.write('vt {} {}\n'.format(vt[i][0],vt[i][1]))

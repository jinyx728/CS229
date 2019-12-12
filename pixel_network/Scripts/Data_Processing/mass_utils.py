######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import numpy as np
import os
from os.path import join
from obj_io import Obj,read_obj,write_obj

def get_m(density,vts,fcs):
    I1,I2,I3=fcs[:,0],fcs[:,1],fcs[:,2]
    v1,v2,v3=vts[I1],vts[I2],vts[I3]
    st=torch.norm(torch.cross(v2-v1,v3-v1),dim=1)
    s=torch.zeros((vts.size(0)),dtype=vts.dtype,device=vts.device)
    s.scatter_add_(0,I1,st)
    s.scatter_add_(0,I2,st)
    s.scatter_add_(0,I3,st)
    s*=density/2/3
    return s.unsqueeze(1)

class MassUtils:
    def __init__(self):
        self.shared_data_dir='../../shared_data_midres'
        self.device=torch.device('cuda:0')
        self.dtype=torch.double
        tshirt_path=join(self.shared_data_dir,'flat_tshirt.obj')
        tshirt_obj=read_obj(tshirt_path)
        vts=torch.from_numpy(tshirt_obj.v).to(device=self.device,dtype=self.dtype)
        fcs=torch.from_numpy(tshirt_obj.f).to(device=self.device,dtype=torch.long)
        self.m=get_m(1,vts,fcs)
        m=self.m.detach().cpu().numpy()
        print('m,min:',np.min(m),'max:',np.max(m))
        np.savetxt(join(self.shared_data_dir,'m.txt'),m)

if __name__=='__main__':
    m=MassUtils()




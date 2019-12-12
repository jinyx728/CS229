######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import numpy as np
import argparse

def calc_uvn(uhat,vhat,nhat,offset):
    u=np.sum(uhat*offset,axis=1).reshape((-1,1))
    v=np.sum(vhat*offset,axis=1).reshape((-1,1))
    n=np.sum(nhat*offset,axis=1).reshape((-1,1))
    uvn=np.hstack([u,v,n])
    return uvn

def get_uvn(uvnhat_dir,uvn_dir,offset_dir,id):
    offset=np.load(os.path.join(offset_dir,'offset_{:08d}.npy'.format(id)))

    front_uhat=np.load(os.path.join(uvnhat_dir,'front_uhats_{:08d}.npy').format(id))
    front_vhat=np.load(os.path.join(uvnhat_dir,'front_vhats_{:08d}.npy').format(id))
    front_nhat=np.load(os.path.join(uvnhat_dir,'front_nhats_{:08d}.npy').format(id))
    front_uvn=calc_uvn(front_uhat,front_vhat,front_nhat,offset)
    np.save(os.path.join(uvn_dir,'front_uvn_offset_{:08d}.npy'.format(id)),front_uvn)
    back_uhat=np.load(os.path.join(uvnhat_dir,'back_uhats_{:08d}.npy').format(id))
    back_vhat=np.load(os.path.join(uvnhat_dir,'back_vhats_{:08d}.npy').format(id))
    back_nhat=np.load(os.path.join(uvnhat_dir,'back_nhats_{:08d}.npy').format(id))
    back_uvn=calc_uvn(back_uhat,back_vhat,back_nhat,offset)
    np.save(os.path.join(uvn_dir,'back_uvn_offset_{:08d}.npy'.format(id)),back_uvn)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-uvnhat_dir',default='/data/zhenglin/poses_v3/midres_skin_tshirt_nuvs')
    parser.add_argument('-uvn_dir',default='/data/zhenglin/poses_v3/midres_skin_tshirt_nuvs')
    parser.add_argument('-offset_dir',default='/data/zhenglin/poses_v3/midres_offset_npys')
    parser.add_argument('-start',type=int,default=0)
    parser.add_argument('-end',type=int,default=15000)
    args=parser.parse_args()

    for i in range(args.start,args.end+1):
        try:
            print(i)
            get_uvn(args.uvnhat_dir,args.uvn_dir,args.offset_dir,i)
        except Exception as e:
            print(e)

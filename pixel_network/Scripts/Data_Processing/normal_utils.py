######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import os
import numpy as np
from obj_io import Obj,read_obj,write_obj
import argparse

# def cross_prod(u,v):
#     '''
#     size: [nBat,nVts,3]
#     '''
#     assert(u.size(2)==3)
#     assert(v.size(2)==3)
#     u1,u2,u3=u[:,:,:1],u[:,:,1:2],u[:,:,2:3]
#     v1,v2,v3=v[:,:,:1],v[:,:,1:2],v[:,:,2:3]
#     w1=u2*v3-u3*v2
#     w2=u3*v1-u1*v3
#     w3=u1*v2-u2*v1
#     w=torch.cat([w1,w2,w3],dim=2)
#     return w

# def cot(u,v):
#     '''
#     size: [nBat,nVts,3]
#     '''
#     eps=1e-10
#     num=torch.sum(u*v,dim=2) # cos
#     denom=torch.norm(cross_prod(u,v),dim=2) # sin
#     denom[denom<eps]=eps # cos/sin
#     return num/denom

def calc_normals(vts,fcs):
    '''
    size: vts: [nBat,nVts,3], fcs: [nFcs,3]
    '''
    device=vts.device
    dtype=vts.dtype
    nBat,nVts=vts.size(0),vts.size(1)
    assert(vts.size(2)==3)
    assert(fcs.size(1)==3)
    fcs=fcs.to(device)
    i1,i2,i3=fcs[:,0],fcs[:,1],fcs[:,2]
    v1,v2,v3=vts[:,i1,:],vts[:,i2,:],vts[:,i3,:]
    # fc_nm=cross_prod(v2-v1,v3-v1)
    fc_nm=torch.cross(v2-v1,v3-v1)

    n=torch.zeros(nBat,nVts,3,dtype=dtype,device=device)
    I1=i1.view(1,-1,1).repeat(nBat,1,3)
    I2=i2.view(1,-1,1).repeat(nBat,1,3)
    I3=i3.view(1,-1,1).repeat(nBat,1,3)

    n.scatter_add_(1,I1,fc_nm)
    n.scatter_add_(1,I2,fc_nm)
    n.scatter_add_(1,I3,fc_nm)

    # safe normalization
    eps=1e-10
    ln=torch.norm(n,dim=2)
    ln=torch.clamp(ln,min=eps)
    # ln[ln<eps]=eps
    n=n/ln.unsqueeze(2)

    return n

def save_obj_with_normals(path,vts,fcs,nms):
    assert(len(vts)==len(nms))
    with open(path,'w') as f:
        nVts=len(vts)
        for i in range(nVts):
            vt=vts[i]
            f.write('v {} {} {}\n'.format(vt[0],vt[1],vt[2]))
        for i in range(nVts):
            vt=vts[i]+nms[i]*0.02
            f.write('v {} {} {}\n'.format(vt[0],vt[1],vt[2]))
        for i in range(nVts):
            nm=nms[i]
            f.write('vn {} {} {}\n'.format(nm[0],nm[1],nm[2]))
        for fc in fcs:
            f.write('f {0}//{0} {1}//{1} {2}//{2}\n'.format(fc[0]+1,fc[1]+1,fc[2]+1))
        for i in range(nVts):
            f.write('l {} {}\n'.format(i+1,i+nVts+1))


class NormalManager:
    def __init__(self,**kwargs):
        # load fcs
        self.shared_data_dir=kwargs['shared_data_dir']
        
        self.data_root_dir=kwargs['data_root_dir']
        
        self.skin_dir=kwargs['skin_dir']
        
        self.cloth_type=kwargs.get('cloth_type', 'lowres_tshirt')
        assert(self.cloth_type in ['lowres_tshirt', 'midres_tshirt', 'tie'])
        self.cloth_prefix='necktie' if self.cloth_type== 'tie' else 'tshirt'
        
        if self.cloth_type == 'lowres_tshirt':
            self.cloth_npys_dir=os.path.join(self.data_root_dir,'lowres_tshirt_npys')
        elif self.cloth_type == 'midres_tshirt':
            self.cloth_npys_dir=os.path.join(self.data_root_dir,'midres_tshirt_npys')
        else:
            self.cloth_npys_dir=os.path.join(self.data_root_dir,'necktie_npys')
        if 'cloth_npys_dir' in kwargs:
            self.cloth_npys_dir=kwargs['cloth_npys_dir']
        
        if self.cloth_type == 'lowres_tshirt' or self.cloth_type == 'midres_tshirt':
            rest_cloth_path=os.path.join(self.shared_data_dir,'flat_tshirt.obj')
        else:
            rest_cloth_path=os.path.join(self.shared_data_dir,'necktie_rest.obj')
        rest_cloth_obj=read_obj(rest_cloth_path)
        self.fcs=torch.from_numpy(rest_cloth_obj.f).long()

        self.out_dir='normal_test' # only used for test purpose

    def normal_cos_dis(self,nms1,nms2):
        return 1-torch.mean(torch.sum(nms1*nms2,dim=2))

    def get_normals(self,vts):
        return calc_normals(vts,self.fcs)

    def test_normal(self,test_id):
        cloth_path=os.path.join(self.cloth_npys_dir,'{}_{:08d}.npy'.format(self.cloth_prefix,test_id))
        vts=np.load(cloth_path)
        vts=torch.from_numpy(vts).unsqueeze(0)
        normals=calc_normals(vts,self.fcs)
        out_path=os.path.join(self.out_dir,'normal_{:08d}.obj'.format(test_id))
        save_obj_with_normals(out_path,vts[0].cpu().numpy(),self.fcs,normals[0].cpu().numpy())

    def test_normal_np(self,test_id):
        cloth_path=os.path.join(self.cloth_npys_dir,'{}_{:08d}.npy'.format(self.cloth_prefix,test_id))
        vts=np.load(cloth_path)
        fcs=self.fcs.numpy()
        nms=calc_normals_np(vts,fcs)
        out_path=os.path.join(self.out_dir,'normal_{:08d}.obj'.format(test_id))
        save_obj_with_normals(out_path,vts,fcs,nms)

    def write_normals_from_id(self,test_id,out_dir):
        cloth_path=os.path.join(self.cloth_npys_dir,'{}_{:08d}.npy'.format(self.cloth_prefix,test_id))
        vts=np.load(cloth_path)
        vts=torch.from_numpy(vts).unsqueeze(0)
        normals=calc_normals(vts,self.fcs)[0]
        out_path=os.path.join(out_dir,'normals_{:08d}.npy'.format(test_id))
        np.save(out_path,normals)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-start',type=int,default=0)
    parser.add_argument('-end',type=int,default=14999)
    args=parser.parse_args()

    kwargs={}
    
#     ## for lowres tshirt on Zhenglin's machine
#     kwargs['shared_data_dir']='../../shared_data'
#     kwargs['data_root_dir']='/data/zhenglin/poses_v3'
#     kwargs['skin_dir']='lowres_skin_tshirt'
#     out_dir='/data/zhenglin/poses_v3/lowres_normal_npys'
    
#     ## for neckties on Jenny's machine
#     kwargs['shared_data_dir']='../../shared_data_necktie'
#     kwargs['data_root_dir']='/data/njin19/poses_necktie'
#     kwargs['skin_dir']=os.path.join(kwargs['data_root_dir'], 'skin_neckties')
#     kwargs['cloth_type']='tie'
#     out_dir=os.path.join(kwargs['data_root_dir'], 'necktie_normal_npys')
    
    ## for lowres tshirt on Jenny's machine
    kwargs['shared_data_dir']='../../shared_data'
    kwargs['data_root_dir']='/data/njin19/poses_v3'
    kwargs['skin_dir']=os.path.join(kwargs['data_root_dir'], 'lowres_skin_npys')
    kwargs['cloth_type']='lowres_tshirt'
    out_dir=os.path.join(kwargs['data_root_dir'], 'lowres_normal_npys')
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    normal_manager=NormalManager(**kwargs)
    for test_id in range(args.start,args.end+1):
        try:
            print(test_id)
            normal_manager.write_normals_from_id(test_id,out_dir)
        except Exception as e:
            print(e)
    # normal_manager.test_normal(10)
    # normal_manager.test_normal_np(10)



# def calc_normals(vts,fcs):
#     '''
#     size: vts: [nBat,nVts,3], fcs: [nFcs,3]
#     '''
#     device=vts.device
#     dtype=vts.dtype
#     nBat,nVts=vts.size(0),vts.size(1)
#     assert(vts.size(2)==3)
#     assert(fcs.size(1)==3)
#     fcs=fcs.to(device)
#     i1,i2,i3=fcs[:,0],fcs[:,1],fcs[:,2]
#     v1,v2,v3=vts[:,i1,:],vts[:,i2,:],vts[:,i3,:]
#     print('v1',v1.size())
#     cot1,cot2,cot3=cot(v2-v1,v3-v1),cot(v3-v2,v1-v2),cot(v1-v3,v2-v3)
#     n=torch.zeros(nBat,nVts,3,dtype=dtype,device=device)
#     I1=i1.view(1,-1,1).repeat(nBat,1,3)
#     I2=i2.view(1,-1,1).repeat(nBat,1,3)
#     I3=i3.view(1,-1,1).repeat(nBat,1,3)

#     cot1v2v3=cot1.unsqueeze(2)*(v2-v3)
#     cot2v3v1=cot2.unsqueeze(2)*(v3-v2)
#     cot3v1v2=cot3.unsqueeze(2)*(v1-v2)
#     cot1v3v2=-cot1v2v3
#     cot2v1v3=-cot2v3v1
#     cot3v2v1=-cot3v1v2
#     n.scatter_add_(1,I1,cot2v3v1)
#     n.scatter_add_(1,I1,cot3v2v1)
#     n.scatter_add_(1,I2,cot3v1v2)
#     n.scatter_add_(1,I2,cot1v3v2)
#     n.scatter_add_(1,I3,cot1v2v3)
#     n.scatter_add_(1,I3,cot2v1v3)

#     # safe normalization
#     eps=1e-10
#     ln=torch.norm(n,dim=2)
#     ln[ln<eps]=eps
#     n=n/ln.unsqueeze(2)

#     return n
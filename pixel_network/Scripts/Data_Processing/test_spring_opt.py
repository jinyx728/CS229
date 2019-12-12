######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Train_Conv_Mix')
import torch
import numpy as np
import os
from os.path import join,isdir,isfile
from obj_io import Obj,read_obj,write_obj
from timeit import timeit
from patch_utils import PatchManager
from spring_opt import SpringOptSystem,NewtonOpt,spring_opt_backward
from spring_opt import gather_data,scatter_data,save_data,load_data
from cvxpy_opt_func import load_opt_data
from spring_opt_func import load_axial_data,get_patch_axial_data,SpringOptModule
import argparse
import matplotlib.pyplot as plt
import time

class SpringOptTest:
    def __init__(self):
        self.data_root_dir='/data/zhenglin/poses_v3'
        self.pd_dir='spring_test/newton'
        self.cr_dir='spring_test/newton'
        self.shared_data_dir='../../shared_data'

        self.device=torch.device('cuda:0')
        self.dtype=torch.double

        self.patch_id=-1
        # self.patch_id=13

        res_ctx={'shared_data_dir':self.shared_data_dir}
        ctx={'dtype':self.dtype,'device':self.device,'patch_id':self.patch_id,'max_num_constraints':-1,'use_spring':True}
        self.res_ctx=res_ctx
        self.ctx=ctx
        m,edges,l0=load_opt_data(res_ctx,ctx)
        k=res_ctx['stiffness']
        harmonic_m=1/(1/m[edges[:,0]]+1/m[edges[:,1]])
        k*=harmonic_m
        print('avg m:',np.mean(m),'avg k:',np.mean(k))

        flat_obj=read_obj(join(self.shared_data_dir,'flat_tshirt.obj'))
        self.f=flat_obj.f
        self.agg_n_vts=len(flat_obj.v)

        if self.patch_id>=0:
            self.patch_manager=PatchManager(shared_data_dir=res_ctx['shared_data_dir'])
            self.patch_vt_ids=self.patch_manager.load_patch_vt_ids(self.patch_id)
            self.patch_fc_ids=self.patch_manager.get_patch_fc_ids(self.patch_vt_ids,self.f)
            self.patch_local_fcs=self.patch_manager.get_patch_local_fcs(self.patch_vt_ids,self.f[self.patch_fc_ids])
            patch_edge_ids=self.patch_manager.get_patch_edge_ids(self.patch_vt_ids,edges)
            m=m[self.patch_vt_ids]
            l0=l0[patch_edge_ids]
            edges=self.patch_manager.get_patch_edges(self.patch_id,edges)
            k=k[patch_edge_ids]
            
            self.pd_dir=join(self.pd_dir,'p{}'.format(self.patch_id))
            self.cr_dir=join(self.cr_dir,'p{}'.format(self.patch_id))
        else:
            self.pd_dir=join(self.pd_dir,'whole')
            self.cr_dir=join(self.cr_dir,'whole')

        self.opt_dir=join(self.pd_dir,'opt')
        if not isdir(self.pd_dir):
            os.makedirs(self.pd_dir)
        if not isdir(self.cr_dir):
            os.makedirs(self.cr_dir)
        if not isdir(self.opt_dir):
            os.makedirs(self.opt_dir)

        self.use_axial_springs=True
        if self.use_axial_springs:
            axial_i,axial_w=load_axial_data(res_ctx)
            if self.patch_id>=0:
                axial_i,axial_w=get_patch_axial_data(self.patch_vt_ids,axial_i,axial_w)
            m0,m1,m2,m3=m[axial_i[:,0]],m[axial_i[:,1]],m[axial_i[:,2]],m[axial_i[:,3]]
            axial_harmonic_m=4/(1/m0+1/m1+1/m2+1/m3)
            axial_k=axial_harmonic_m*1e-1

        # m*=2
        # m*=0.05
        m*=0.2

        # out_dir='spring_test/data'
        # if not isdir(out_dir):
        #     os.makedirs(out_dir)
        # np.savetxt(join(out_dir,'stiffen_anchor.txt'),m,fmt='%.100f')
        # np.savetxt(join(out_dir,'edges.txt'),edges,fmt='%d')
        # np.savetxt(join(out_dir,'l0.txt'),l0,fmt='%.100f')
        # np.savetxt(join(out_dir,'k.txt'),k,fmt='%.100f')
        # np.savetxt(join(out_dir,'axial_i.txt'),axial_i,fmt='%d')
        # np.savetxt(join(out_dir,'axial_w.txt'),axial_w,fmt='%.100f')
        # np.savetxt(join(out_dir,'axial_k.txt'),axial_k,fmt='%.100f')
        # exit(0)

        # self.m=torch.ones((len(m),1)).to(dtype=self.dtype,device=self.device)
        # self.m=(torch.ones((len(m),1))+(torch.rand((len(m),1))-0.5)*1).to(dtype=self.dtype,device=self.device)*4
        self.stiffen_anchors_net=torch.from_numpy(m).to(dtype=self.dtype,device=self.device).view(-1,1)/2
        self.stiffen_anchors_reg=torch.from_numpy(m).to(dtype=self.dtype,device=self.device).view(-1,1)/2
        self.edges=torch.from_numpy(edges).to(dtype=torch.long,device=self.device).view(-1,2)
        self.l0=torch.from_numpy(l0).to(dtype=self.dtype,device=self.device).view(-1,1)
        self.k=torch.from_numpy(k).to(dtype=self.dtype,device=self.device).view(-1,1)

        if self.use_axial_springs:
            axial_i=torch.from_numpy(axial_i).to(dtype=torch.long,device=self.device).view(-1,4)
            axial_w=torch.from_numpy(axial_w).to(dtype=self.dtype,device=self.device).view(-1,4)
            axial_k=torch.from_numpy(axial_k).to(dtype=self.dtype,device=self.device).view(-1,1)
            self.axial_data=(axial_i,axial_w,axial_k)
        else:
            self.axial_data=None

    def read_obj(self,path,patch_id=-1):
        pd_obj=read_obj(path)
        pd_vt=pd_obj.v
        f=pd_obj.f
        if patch_id>=0:
            pd_vt=pd_vt[self.patch_vt_ids]
            f=f[self.patch_fc_ids]
        return pd_vt,f

    def write_obj(self,v,f,out_path,patch_id=-1):
        if patch_id>=0:
            full_v=np.zeros((self.agg_n_vts,3))
            full_v[self.patch_vt_ids,:]=v
            v=full_v
        print('write to',out_path)
        write_obj(Obj(v=v,f=f),out_path)


    def test_forward(self,sample_id,n_iters=10):
        # gt_path=join(self.pd_dir,'gt_{:08d}.obj'.format(sample_id))
        # v,f=self.read_obj(gt_path,patch_id=self.patch_id)   
        # print('save gt')
        # np.savetxt('spring_test/data/gt_{:08d}.txt'.format(sample_id),v,fmt='%.100f')
        # exit(0)

        pd_path=join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id))
        v,f=self.read_obj(pd_path,patch_id=self.patch_id)        
        # np.savetxt('spring_test/data/fcs.txt',f,fmt='%d')
        # np.savetxt('spring_test/data/anchor.txt',v,fmt='%.100f')
        v=torch.from_numpy(v).to(device=self.device,dtype=self.dtype)
        system=SpringOptSystem(self.stiffen_anchors_net,self.stiffen_anchors_reg,self.edges,self.l0,self.k,m_alpha=0.1,axial_data=self.axial_data)
        opt=NewtonOpt(system,newton_tol=1e-3,cg_tol=1e-3,cg_max_iter=1000)
        x=v
        start_time=time.time()
        for i in range(n_iters):
            x,data,success=opt.solve(v,x)
            # print('diff',torch.norm(x-v).item())
            end_time=time.time()
            print('forward time:',end_time-start_time)
            if i%1==0:
                x_save=x.detach().cpu().numpy()
                cr_path=join(self.cr_dir,'cr_{:08d}_i{:02d}.obj'.format(sample_id,i))
                self.write_obj(x_save,f,cr_path,patch_id=self.patch_id)
        cr_path=join(self.cr_dir,'cr_{:08d}.npy'.format(sample_id))        
        np.save(cr_path,x_save)
        # m_path=join(self.cr_dir,'m_{:08d}.npy'.format(sample_id))
        # np.save(m_path,data['m_adjusted'].cpu().numpy())

        # data_dir=join(self.cr_dir,'data')
        # if not isdir(data_dir):
        #     os.makedirs(data_dir)
        # save_data(data_dir,gather_data([data]))

    def test_dataset(self,start,end,n_iters=10):
        gt_offset_dir=join(self.data_root_dir,'lowres_offset_npys')
        skin_dir=join(self.data_root_dir,'lowres_skin_npys')
        out_dir=join(self.data_root_dir,'lowres_offsets_i{}'.format(n_iters))
        if not isdir(out_dir):
            os.makedirs(out_dir)
        system=SpringOptSystem(self.m,self.edges,self.l0,self.k,m_alpha=0.1,axial_data=self.axial_data)
        opt=NewtonOpt(system,newton_tol=1e-12,cg_tol=1e-3,cg_max_iter=250)
        for sample_id in range(start,end+1):
            print('sample_id',sample_id)
            try:
                offset=np.load(join(gt_offset_dir,'offset_{:08d}.npy'.format(sample_id)))
                skin=np.load(join(skin_dir,'skin_{:08d}.npy'.format(sample_id)))
                x=torch.from_numpy(offset+skin).to(device=self.device,dtype=self.dtype)
                for i in range(n_iters):
                    x=opt.solve(x)
                x_save=x.cpu().numpy()
                offset_save=x_save-skin
                np.save(join(out_dir,'offset_{:08d}.npy'.format(sample_id)),offset_save)
                # cr_path=join(self.cr_dir,'cr_{:08d}.obj'.format(sample_id))
                # self.write_obj(x_save,None,cr_path,patch_id=self.patch_id)
            except:
                continue

    def test_forward_dir(self,in_dir,out_dir,n_iters=1,start=0,end=2247):
        system=SpringOptSystem(self.stiffen_anchors_net,self.stiffen_anchors_reg,self.edges,self.l0,self.k,m_alpha=0.1,axial_data=self.axial_data)
        opt=NewtonOpt(system,newton_tol=1e-12,cg_tol=1e-3,cg_max_iter=250)
        if not isdir(out_dir):
            os.makedirs(out_dir)

        for sample_id in range(start,end+1):
            pd_path=join(in_dir,'{:08d}.obj'.format(sample_id))
            v,f=self.read_obj(pd_path,patch_id=self.patch_id)
            v=torch.from_numpy(v).to(device=self.device,dtype=self.dtype)
            x=v
            start_time=time.time()
            for i in range(n_iters):
                x,data,success=opt.solve(v,x)
                end_time=time.time()
                print('forward time:',end_time-start_time)
            x_save=x.detach().cpu().numpy()
            cr_path=join(out_dir,'{:08d}.obj'.format(sample_id))
            self.write_obj(x_save,f,cr_path,patch_id=self.patch_id)

    def test_backward(self,sample_id):
        pd_path=join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id))
        pd_v,f=self.read_obj(pd_path,patch_id=self.patch_id)
        pd_v=torch.from_numpy(pd_v).to(device=self.device,dtype=self.dtype)
        gt_path=join(self.pd_dir,'gt_{:08d}.obj'.format(sample_id))
        gt_v,_=self.read_obj(gt_path,patch_id=self.patch_id)
        gt_v=torch.from_numpy(gt_v).to(device=self.device,dtype=self.dtype)
        dv=pd_v-gt_v
        cr_path=join(self.cr_dir,'cr_{:08d}.npy'.format(sample_id))
        cr_v=np.load(cr_path)
        cr_v=torch.from_numpy(cr_v).to(device=self.device,dtype=self.dtype)
        # m_path=join(self.cr_dir,'m_{:08d}.npy'.format(sample_id))
        # m_adjusted=np.load(m_path)
        # m_adjusted=torch.from_numpy(m_adjusted).to(device=self.device,dtype=self.dtype)

        # system=SpringOptSystem(self.m,self.edges,self.l0,self.k,m_alpha=0.1,axial_data=self.axial_data)
        system=SpringOptSystem(self.stiffen_anchors_net,self.stiffen_anchors_reg,self.edges,self.l0,self.k,m_alpha=0.1,axial_data=self.axial_data)
        system.use_m_adjusted=False
        data=system.get_data(cr_v)
        data['c']=pd_v
        data['anchors_net']=pd_v
        data['anchors_reg']=pd_v
        data['stiffen_anchors_net']=self.stiffen_anchors_net
        data['stiffen_anchors_reg']=self.stiffen_anchors_reg

        # data['m_adjusted']=m_adjusted
        J=system.get_J(data)
        norm_J=torch.norm(J)
        data['J_rms']=norm_J/np.sqrt(len(cr_v))

        dx=spring_opt_backward(system,data,dv,cg_tol=1e-3,cg_max_iter=250)
        grad_path=join(self.cr_dir,'grad_{:08d}.npy'.format(sample_id))
        print('save to',grad_path)
        np.save(grad_path,dx.cpu().numpy())

    def test_grad(self,sample_id):
        pd_path=join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id))
        print('pd_path',pd_path)
        pd_vt,_=self.read_obj(pd_path,patch_id=self.patch_id)
        grad=np.load(join(self.cr_dir,'grad_{:08d}.npy'.format(sample_id)))
        print('grad.norm',np.linalg.norm(grad))
        grad_len=1
        ed_vt=pd_vt-grad*grad_len
        n_vts=len(pd_vt)
        obj_path=join(self.cr_dir,'grad_{:08d}.obj'.format(sample_id))
        print('write to',obj_path)
        with open(obj_path,'w') as f:
            for v in pd_vt:
                f.write('v {} {} {}\n'.format(v[0],v[1],v[2]))
            for v in ed_vt:
                f.write('v {} {} {}\n'.format(v[0],v[1],v[2]))
            for i in range(n_vts):
                f.write('l {} {}\n'.format(i+1,i+1+n_vts))

    def test_module(self,sample_id,n_iters=1):
        pd_path=join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id))
        pd_v,f=self.read_obj(pd_path,patch_id=self.patch_id)
        pd_v=torch.from_numpy(pd_v).to(device=self.device,dtype=self.dtype).unsqueeze(0)
        gt_path=join(self.pd_dir,'gt_{:08d}.obj'.format(sample_id))
        gt_v,_=self.read_obj(gt_path,patch_id=self.patch_id)
        gt_v=torch.from_numpy(gt_v).to(device=self.device,dtype=self.dtype).unsqueeze(0)
        proj_module=SpringOptModule(self.res_ctx,self.ctx)
        x=pd_v
        x.requires_grad_(True)
        for i in range(n_iters):
            x=proj_module(x)
        loss=torch.sum((gt_v-x)**2)/2
        loss.backward()
        print('grad.norm',torch.norm(pd_v.grad))

    def test_loss_along_line(self,sample_id,n_iters):
        pd_path=join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id))
        pd_v,f=self.read_obj(pd_path,patch_id=self.patch_id)
        pd_v=torch.from_numpy(pd_v).to(device=self.device,dtype=self.dtype).unsqueeze(0)

        # gt_vt=np.load(join(self.data_root_dir,'lowres_skin_npys/skin_{:08d}.npy'.format(sample_id)))+np.load(join(self.data_root_dir,'lowres_offsets_i10/offset_{:08d}.npy'.format(sample_id)))
        gt_path=join(self.pd_dir,'gt_{:08d}.obj'.format(sample_id))
        gt_v,f=self.read_obj(gt_path,patch_id=self.patch_id)
        gt_v=torch.from_numpy(gt_v).to(device=self.device,dtype=self.dtype).unsqueeze(0)

        proj_module=SpringOptModule(self.res_ctx,self.ctx)

        def f(x):
            for i in range(n_iters):
                x=proj_module(x)
            return torch.sum(((x-gt_v)**2).view(x.size(0),-1),dim=1)/2

        pd_v.requires_grad_(True)
        loss=f(pd_v)
        loss.backward()
        g=pd_v.grad[0]
        pd_v.requires_grad_(False)

        loss_list=[]
        total_n=100
        processed_n=0
        end=2
        batch_size=1
        while processed_n<total_n:
            x=pd_v.repeat(batch_size,1,1)
            for i in range(batch_size):
                t=(i+processed_n)/total_n*end
                x[i]-=t*g
            loss=f(x)
            loss_list+=loss.tolist()
            processed_n+=batch_size
        print(loss_list)
        np.savetxt(join(self.opt_dir,'loss_{}.txt'.format(end)),np.array(loss_list))

    def plot_loss_along_line(self,sample_id):
        end=2
        loss=np.loadtxt(join(self.opt_dir,'loss_{}.txt'.format(2)))
        x=np.linspace(0,end,len(loss))
        fig=plt.gcf()
        ax=plt.gca()
        ax.plot(x,loss)
        ax.set_title('iter=10')
        plot_path=join(self.opt_dir,'loss_{}.png'.format(end))
        print('plot_path',plot_path)
        fig.savefig(plot_path)



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-start',type=int,default=0)
    parser.add_argument('-end',type=int,default=0)
    args=parser.parse_args()

    test=SpringOptTest()
    # test.test_forward(106,n_iters=1)
    # test.test_dataset(args.start,args.end)
    test.test_backward(106)
    test.test_grad(106)
    # test.test_module(106,n_iters=10)
    # test.test_loss_along_line(106,n_iters=10)
    # test.plot_loss_along_line(106)
    # test.test_forward_dir('/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/joint_data/seq1/videos/collected_objs','/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/joint_data/seq1/videos/corrected_objs',start=args.start,end=args.end)
    # test.test_forward_dir('/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/joint_data/seq2/videos/collected_objs','/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/joint_data/seq2/videos/corrected_objs',start=args.start,end=args.end)


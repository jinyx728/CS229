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
import cvx_opt_cpp
from cvxpy_opt_func import load_opt_data
from ecos_opt_func import init_ecos_opt_module
import matplotlib.pyplot as plt
from timeit import timeit
from scipy.sparse import csr_matrix
from patch_utils import PatchManager
import time

class CvxOptSimpleTest:
    def __init__(self):
        dtype=torch.double
        m=torch.tensor([1,1],dtype=dtype)
        edges=torch.tensor([[0,1]],dtype=torch.long)
        l0=torch.tensor([1],dtype=dtype)
        cvx_opt_cpp.init(m,edges,l0,2)
        self.dtype=dtype

    def test(self):
        tgt_x=torch.tensor([[0,0,0,2,0,0]],dtype=self.dtype)
        x=cvx_opt_cpp.solve(tgt_x)
        print(x)

@timeit
def back_track_line_search(x0,dx,df,f,fx0,alpha=0.1,beta=0.5):
    dfdx=torch.sum(df*dx)
    t=1
    loss=fx0
    n_steps=0
    while True: 
        loss=f(x0+t*dx)
        if loss<=fx0+alpha*t*dfdx:
            break
        print('n_steps',n_steps,'t',t,'loss',loss.item())
        t*=beta
        n_steps+=1
        if n_steps>64:
            print('not a decent direction')
            assert(False)            
    return t,loss

class CvxOptCppTest:
    def __init__(self):
        self.data_root_dir='/data/zhenglin/poses_v3'
        self.shared_data_dir='../../shared_data'
        # self.shared_data_dir='../../shared_data_midres'
        self.pd_dir='spring_test/cvx_opt_cpp'
        self.cr_dir='spring_test/cvx_opt_cpp'
        self.opt_dir='spring_test/cvx_opt_cpp'

        # self.cr_dir='opt_test/cpp_test'
        # self.opt_dir='opt_test/cpp_test/opt'
        # self.gt_offset_dir=join(self.data_root_dir,'lowres_offset_npys')
        # self.cr_gt_offset_dir=join(self.data_root_dir,'lowres_offsets_len-2lap1')
        self.skin_dir=join(self.data_root_dir,'lowres_skin_npys')
        # self.pd_dir='../../rundir/lowres/xyz/eval_test'
        # self.pd_dir='../../rundir/lowres_ecos/xyz/eval_test'
        # self.pd_dir='../../rundir/lowres/xyz/eval_train'

        # self.cr_dir='opt_test/cpp_test_midres/cr'
        # self.opt_dir='opt_test/cpp_test_midres/opt'
        # self.shared_data_dir='../../shared_data_midres'
        # self.gt_offset_dir=join(self.data_root_dir,'midres_offset_npys')
        # self.cr_gt_offset_dir=join(self.data_root_dir,'midres_offsets_len-2lap-1')
        # self.skin_dir=join(self.data_root_dir,'midres_skin_npys')
        # self.pd_dir='../../rundir/midres/xyz/eval_test'

        self.plot_dir='opt_test/cpp_test/figs'
        # if not isdir(self.cr_dir):
        #     os.makedirs(self.cr_dir)
        # if not isdir(self.plot_dir):
        #     os.makedirs(self.plot_dir)
        # if not isdir(self.opt_dir):
        #     os.makedirs(self.opt_dir)

        self.device=torch.device('cpu')
        self.dtype=torch.double
        # self.batch_size=40
        self.batch_size=40

        self.patch_id=-1
        # self.patch_id=13
        # self.verbose=True
        self.verbose=False
        self.use_debug=False

        # self.use_lap=False
        self.use_lap=False

        self.use_spring=False
        self.use_variable_m=False

        tol=1e-6
        tol_inacc=5e-3
        reltol=5e-1
        reltol_inacc=5e0
        # tol=1e-8
        # tol_inacc=1e-4
        # reltol=1e-8
        # reltol_inacc=1e-4
        self.tol=tol
        self.reltol=reltol

        res_ctx={'shared_data_dir':self.shared_data_dir}
        ctx={'dtype':self.dtype,'batch_size':self.batch_size,'max_num_constraints':-1,'use_lap':self.use_lap,'lmd_lap':1e-1,'patch_id':self.patch_id,'verbose':self.verbose,'use_debug':self.use_debug,'use_spring':self.use_spring,'lmd_k':5e-2,'maxit':50,'feastol':tol,'abstol':tol,'reltol':reltol,'feastol_inacc':tol_inacc*2,'abstol_inacc':tol_inacc,'reltol_inacc':reltol_inacc}
        # m,edges,l0=load_opt_data({'shared_data_dir':self.shared_data_dir},{'max_num_constraints':-1})

        # m=torch.from_numpy(m).to(device=self.device,dtype=self.dtype)
        # edges=torch.from_numpy(edges).to(device=self.device,dtype=torch.long)
        # l0=torch.from_numpy(l0).to(device=self.device,dtype=self.dtype)

        # cvx_opt_cpp.init(m,edges,l0,self.batch_size)
        flat_obj=read_obj(join(self.shared_data_dir,'flat_tshirt.obj'))
        self.f=flat_obj.f
        self.agg_n_vts=len(flat_obj.v)

        self.proj_module=init_ecos_opt_module(res_ctx,ctx)
        if self.use_lap:
            self.L=self.proj_module.L.tocsr()

        if self.patch_id>=0:
            self.patch_manager=PatchManager(shared_data_dir=res_ctx['shared_data_dir'])
            self.patch_vt_ids=self.patch_manager.load_patch_vt_ids(self.patch_id)
            self.patch_fc_ids=self.patch_manager.get_patch_fc_ids(self.patch_vt_ids,self.f)
            self.patch_local_fcs=self.patch_manager.get_patch_local_fcs(self.patch_vt_ids,self.f[self.patch_fc_ids])
            self.cr_dir=join(self.cr_dir,'p{:02d}'.format(self.patch_id))
            self.pd_dir=join(self.pd_dir,'p{:02d}'.format(self.patch_id))
            self.opt_dir=join(self.opt_dir,'p{:02d}'.format(self.patch_id))
        else:
            self.cr_dir=join(self.cr_dir,'whole')
            self.pd_dir=join(self.pd_dir,'whole')
            self.opt_dir=join(self.opt_dir,'whole')

            # self.opt_dir=join(self.cr_dir,'opt')
            # self.opt_dir=join(self.cr_dir,'debug')
        if not isdir(self.cr_dir):
            os.makedirs(self.cr_dir)
        if not isdir(self.opt_dir):
            os.makedirs(self.opt_dir)
        if not isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

    def read_obj(self,path,patch_id=-1):
        pd_obj=read_obj(path)
        pd_vt=pd_obj.v
        f=pd_obj.f
        if patch_id>=0:
            pd_vt=pd_vt[self.patch_vt_ids]
            f=f[self.patch_fc_ids]
        return pd_vt,f

    def write_obj(self,v,f,prefix,sample_id,patch_id=-1):
        if patch_id>=0:
            full_v=np.zeros((self.agg_n_vts,3))
            full_v[self.patch_vt_ids,:]=v
            v=full_v
        out_path='{}_{:08d}.obj'.format(prefix,sample_id)
        print('write to',out_path)
        write_obj(Obj(v=v,f=f),out_path)

    def test(self,sample_id):
        pd_vt,f=self.read_obj(join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id)),patch_id=self.patch_id)
        pd_vts=torch.from_numpy(pd_vt).to(device=self.device,dtype=self.dtype).unsqueeze(0)
        pd_vts=pd_vts.repeat(self.batch_size,1,1)
        tgt_x=pd_vts

        # x,y,z,s=self.proj_module(tgt_x)
        if self.use_variable_m:
            m=torch.from_numpy(self.proj_module.m).to(device=self.device,dtype=self.dtype).unsqueeze(0)
            x,y,z,s,success=cvx_opt_cpp.solve_forward_variable_m(tgt_x,m)
        else:
            x,y,z,s,success=cvx_opt_cpp.solve_forward(tgt_x)

        np.save(join(self.cr_dir,'x_{:08d}.npy'.format(sample_id)),x.cpu().numpy())
        np.save(join(self.cr_dir,'y_{:08d}.npy'.format(sample_id)),y.cpu().numpy())
        np.save(join(self.cr_dir,'z_{:08d}.npy'.format(sample_id)),z.cpu().numpy())
        np.save(join(self.cr_dir,'s_{:08d}.npy'.format(sample_id)),s.cpu().numpy())
        np.save(join(self.cr_dir,'success_{:08d}.npy'.format(sample_id)),success.cpu().numpy())
        if self.use_variable_m:
            np.save(join(self.cr_dir,'m_{:08d}.npy'.format(sample_id)),m.cpu().numpy())
        n_vts=len(pd_vt)
        cr_vt=x[0,:n_vts*3].view(-1,3).cpu().numpy()

        self.write_obj(cr_vt,f,prefix=join(self.cr_dir,'cr'),sample_id=sample_id,patch_id=self.patch_id)
        # self.write_obj(cr_vt,f,prefix=join(self.opt_dir,'cr_tgt_015'),sample_id=sample_id,patch_id=self.patch_id)
        # self.write_obj(cr_vt,f,prefix=join(self.opt_dir,'cr_gt'),sample_id=sample_id,patch_id=self.patch_id)

    def test_samples_dir(self,samples_dir):
        sample_dirs=os.listdir(samples_dir)
        pd_pattern='pd_cloth_{:08d}.obj'
        cr_pattern='cr_ineq_cloth_{:08d}.obj'
        # pd_pattern='pd_{:08d}.obj'
        # cr_pattern='cr_ineq_{:08d}.obj'

        anchor=[]
        ids=[]
        for i,sample_dir in enumerate(sample_dirs):
            sample_id=int(sample_dir)
            pd_path=join(samples_dir,sample_dir,pd_pattern.format(sample_id))
            pd_obj=read_obj(pd_path)
            v,f=pd_obj.v,pd_obj.f
            v=torch.from_numpy(v).to(device=self.device,dtype=self.dtype)
            v=v.unsqueeze(0)
            anchor.append(v)
            ids.append(sample_id)
            if len(anchor)==self.batch_size or i==len(sample_dirs)-1:
                anchor=torch.cat(anchor,dim=0)
                start_time=time.time()
                x=self.proj_module(anchor)
                end_time=time.time()
                # print('ellapse:',end_time-start_time)
                for j,sample_id in enumerate(ids):
                    x_save=x[j].detach().cpu().numpy()
                    cr_path=join(samples_dir,'{:08d}'.format(sample_id),cr_pattern.format(sample_id))
                    print('write to',cr_path)
                    write_obj(Obj(v=x_save,f=f),cr_path)
                anchor=[]
                ids=[]
            # break

    def avg_samples_dir(self,samples_dir):
        sample_dirs=os.listdir(samples_dir)
        pd_pattern='pd_cloth_{:08d}.obj'
        cr_pattern='cr_cloth_{:08d}.obj'
        avg_pattern='avg_cloth_{:08d}.obj'

        for i,sample_dir in enumerate(sample_dirs):
            sample_id=int(sample_dir)
            pd_path=join(samples_dir,sample_dir,pd_pattern.format(sample_id))
            pd_obj=read_obj(pd_path)
            v0,f=pd_obj.v,pd_obj.f
            cr_path=join(samples_dir,sample_dir,cr_pattern.format(sample_id))
            cr_obj=read_obj(cr_path)
            v1=cr_obj.v
            avg_path=join(samples_dir,sample_dir,avg_pattern.format(sample_id))
            print('write to',avg_path)
            write_obj(Obj(v=(v0+v1)/2,f=f),avg_path)


    def test_offset_dir(self):
        in_offset_dir=join(self.data_root_dir,'lowres_offset_npys')
        out_offset_dir=join(self.data_root_dir,'lowres_offsets_len-2')
        skin_dir=join(self.data_root_dir,'lowres_skin_npys')
        if not isdir(out_offset_dir):
            os.makedirs(out_offset_dir)
        anchors=[]
        ids=[]
        skins=[]
        def flush():
            if len(anchors)==0:
                return
            x=torch.cat(anchors,dim=0)
            x=self.proj_module(x)
            for i,sample_id in enumerate(ids):
                out_offset=x[i].detach().cpu().numpy()-skins[i]
                out_offset_path=join(out_offset_dir,'offset_{:08d}.npy'.format(sample_id))
                print('write to',out_offset_path)
                np.save(out_offset_path,out_offset)
            anchors.clear()
            ids.clear()
            skins.clear()

        for sample_id in range(17317,30000):
            in_offset_path=join(in_offset_dir,'offset_{:08d}.npy'.format(sample_id))
            skin_path=join(skin_dir,'skin_{:08d}.npy'.format(sample_id))
            if not isfile(in_offset_path):
                print('not found',in_offset_path)
                continue
            if not isfile(skin_path):
                print('not found',skin_path)
                continue
            in_offset=np.load(in_offset_path)
            skin=np.load(skin_path)
            v=torch.from_numpy(in_offset+skin).to(device=self.device,dtype=self.dtype)
            v=v.unsqueeze(0)
            anchors.append(v)
            ids.append(sample_id)
            skins.append(skin)
            if len(anchors)==self.batch_size:
                flush()
        flush()


    def test_loss(self,sample_id):
        pd_vt,f=self.read_obj(join(self.pd_dir,'{:08d}/pd_cloth_{:08d}.obj'.format(sample_id,sample_id)),patch_id=self.patch_id)
        cr_gt_vt,_=self.read_obj(join('opt_test/cpp_test/cr_gt_{:08d}_bak.obj'.format(sample_id)),patch_id=self.patch_id)
        agg_tgt_vt,_=self.read_obj(join('opt_test/cpp_test/opt/cr_tgt_{:08d}_lap0.obj'.format(sample_id)),patch_id=self.patch_id)
        tgt_vt=read_obj(join(self.opt_dir,'tgt_015_{:08d}.obj'.format(sample_id))).v[self.patch_vt_ids]
        print(np.linalg.norm(tgt_vt-cr_gt_vt,axis=1))
        print(np.linalg.norm(agg_tgt_vt-cr_gt_vt,axis=1))
        print('agg Ninf:',np.max(np.linalg.norm(agg_tgt_vt-cr_gt_vt,axis=1)),'N2:',np.linalg.norm(agg_tgt_vt-cr_gt_vt))
        print('patch Ninf:',np.max(np.linalg.norm(tgt_vt-cr_gt_vt,axis=1)),'N2:',np.linalg.norm(tgt_vt-cr_gt_vt))
        self.write_obj(cr_gt_vt,f,prefix=join(self.opt_dir,'cr_gt'),sample_id=sample_id,patch_id=self.patch_id)
        self.write_obj(agg_tgt_vt,f,prefix=join(self.opt_dir,'agg_tgt'),sample_id=sample_id,patch_id=self.patch_id)


    def test_backward(self,sample_id):
        # pd_obj=read_obj(join(self.pd_dir,'{:08d}/pd_cloth_{:08d}.obj'.format(sample_id,sample_id)))
        # pd_obj=read_obj(join(self.pd_dir,'{:08d}/gt_cloth_{:08d}.obj'.format(sample_id,sample_id)))
        # pd_vt=np.load(join(self.pd_dir,'{:08d}/pd_vts_{:08d}.npy'.format(sample_id,sample_id)))
        pd_vt,f=self.read_obj(join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id)),patch_id=self.patch_id)

        # sample_id=0
        # pd_vt=np.loadtxt(join(self.opt_dir,'forward_tgt_0_12.txt')).reshape(-1,3)

        # pd_vt=pd_vt[self.patch_vt_ids]

        pd_vts=torch.from_numpy(pd_vt).to(device=self.device,dtype=self.dtype).unsqueeze(0)
        pd_vts=pd_vts.repeat(self.batch_size,1,1)
        tgt_x=pd_vts

        # cr_vt=read_obj(join(self.cr_dir,'dbg_cr_pd_{:08d}.obj'.format(sample_id))).v
        cr_vt=read_obj(join(self.cr_dir,'cr_{:08d}.obj'.format(sample_id))).v
        if self.patch_id>=0:
            cr_vt=cr_vt[self.patch_vt_ids]
        cr_vts=torch.from_numpy(cr_vt).to(device=self.device,dtype=self.dtype).unsqueeze(0)
        cr_vts=cr_vts.repeat(self.batch_size,1,1)

        # gt_vt,_=self.read_obj(join(self.pd_dir,'{:08}/gt_cloth_{:08d}.obj'.format(sample_id,sample_id)),patch_id=self.patch_id)
        gt_offset_dir=join(self.data_root_dir,'lowres_offsets_len-2')
        gt_vt=np.load(join(gt_offset_dir,'offset_{:08d}.npy'.format(sample_id)))+np.load(join(self.skin_dir,'skin_{:08d}.npy'.format(sample_id)))
        if self.patch_id>=0:
            gt_vt=gt_vt[self.patch_vt_ids]
        gt_vts=torch.from_numpy(gt_vt).to(device=self.device,dtype=self.dtype).unsqueeze(0)
        gt_vts=gt_vts.repeat(self.batch_size,1,1)

        n_vts=len(gt_vt)
        grad=(cr_vts-gt_vts)/(self.batch_size)

        x=np.load(join(self.cr_dir,'x_{:08d}.npy'.format(sample_id)))
        x=torch.from_numpy(x).to(device=self.device,dtype=self.dtype)
        y=np.load(join(self.cr_dir,'y_{:08d}.npy'.format(sample_id)))
        y=torch.from_numpy(y).to(device=self.device,dtype=self.dtype)
        z=np.load(join(self.cr_dir,'z_{:08d}.npy'.format(sample_id)))
        z=torch.from_numpy(z).to(device=self.device,dtype=self.dtype)
        s=np.load(join(self.cr_dir,'s_{:08d}.npy'.format(sample_id)))
        s=torch.from_numpy(s).to(device=self.device,dtype=self.dtype)
        success=np.load(join(self.cr_dir,'success_{:08d}.npy'.format(sample_id)))
        success=torch.from_numpy(success).to(device=self.device,dtype=torch.long)

        if self.use_variable_m:
            m=np.load(join(self.cr_dir,'m_{:08d}.npy'.format(sample_id)))
            m=torch.from_numpy(m).to(device=self.device,dtype=self.dtype).unsqueeze(0)
            out_grads,out_m_grads=cvx_opt_cpp.solve_backward_variable_m(grad,tgt_x,m,[x,y,z,s,success])
        else:
            out_grads=cvx_opt_cpp.solve_backward(grad,tgt_x,[x,y,z,s,success])

        grad_path=join(self.cr_dir,'grad_{:08d}.npy'.format(sample_id))
        print('write to',grad_path)
        out_grad=out_grads[0].cpu().numpy()
        if np.any(np.isnan(out_grad)):
            print('grad has nan')
        np.save(grad_path,out_grad)

        if self.use_variable_m:
            grad_m_path=join(self.cr_dir,'grad_m_{:08d}.npy'.format(sample_id))
            print('write to',grad_m_path)
            np.save(grad_m_path,out_m_grads[0].cpu().numpy())


    def test_grad(self,sample_id):
        pd_vt,_=self.read_obj(join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id)),patch_id=self.patch_id)
        grad=np.load(join(self.cr_dir,'grad_{:08d}.npy'.format(sample_id))).reshape(pd_vt.shape)
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

    def test_opt(self,sample_id):
        pd_vt,fcs=self.read_obj(join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id)),patch_id=self.patch_id)
        pd_vts=torch.from_numpy(pd_vt).to(device=self.device,dtype=self.dtype).unsqueeze(0)
        pd_vts=pd_vts.repeat(self.batch_size,1,1)
        tgt_x=pd_vts

        gt_vt,_=self.read_obj(join(self.cr_dir,'gt_{:08d}.obj'.format(sample_id)),patch_id=self.patch_id)
        if self.patch_id>=0:
            gt_vt=gt_vt[self.patch_vt_ids]
        gt_vts=torch.from_numpy(gt_vt).to(device=self.device,dtype=self.dtype).unsqueeze(0)
        gt_vts=gt_vts.repeat(self.batch_size,1,1)  

        def f(x):
            x=self.proj_module(x)
            # return torch.sum((x-gt_vts)**2)/(x.size(0)*x.size(1))
            return torch.sum((x-gt_vts)**2)/2

        counter=0
        loss=1
        while loss>1e-12:
            tgt_x.requires_grad_(True)
            loss=f(tgt_x)
            loss.backward()
            grad=tgt_x.grad.clone()
            with torch.no_grad():
                step,loss=back_track_line_search(tgt_x,-grad,grad,f,loss.item())
                tgt_x.requires_grad_(False)
            tgt_x-=grad*step
            print('-> counter',counter,'step_size:',step,'loss',loss.item(),'grad',torch.norm(grad).item())
            self.write_obj(tgt_x.cpu().numpy()[0],fcs,prefix=join(self.opt_dir,'tgt_{:03d}'.format(counter)),sample_id=sample_id,patch_id=self.patch_id)
            counter+=1

    def test_plot(self,sample_id):
        pd_vt,_=self.read_obj(join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id)),patch_id=self.patch_id)
        pd_vts=torch.from_numpy(pd_vt).to(device=self.device,dtype=self.dtype).unsqueeze(0)
        gt_vt,_=self.read_obj(join(self.cr_dir,'gt_{:08d}.obj'.format(sample_id)),patch_id=self.patch_id)
        if self.patch_id>=0:
            gt_vt=gt_vt[self.patch_vt_ids]
        gt_vts=torch.from_numpy(gt_vt).to(device=self.device,dtype=self.dtype).unsqueeze(0)
        gt_vts=gt_vts.repeat(self.batch_size,1,1)  

        def f(x):
            x=self.proj_module(x)
            # return torch.sum(((x-gt_vts)**2).view(x.size(0),-1),dim=1)/x.size(1)
            return torch.sum(((x-gt_vts)**2).view(x.size(0),-1),dim=1)/2

        grad_path=join(self.cr_dir,'grad_{:08d}.npy'.format(sample_id))
        if not isfile(grad_path):
        # if True:
            print(grad_path,' does not exist, compute')
            tgt_x=pd_vts # only 1 sample
            tgt_x.requires_grad_(True)
            loss=f(tgt_x)[0]
            loss.backward()
            grad=tgt_x.grad.clone()[0]
            # np.save(grad_path,grad.cpu().numpy())
        else:
            grad=torch.from_numpy(np.load(grad_path)).view(-1,3)

        loss_list=[]
        total_n=40
        processed_n=0
        end=2
        while processed_n<total_n:
            tgt_x=pd_vts.repeat(self.batch_size,1,1)
            tgt_x=tgt_x.detach()
            for i in range(self.batch_size):
                t=(i+processed_n)/total_n*end
                tgt_x[i]-=t*grad
            loss=f(tgt_x)
            loss_list+=loss.tolist()
            processed_n+=self.batch_size
        print(loss_list)
        np.savetxt(join(self.cr_dir,'loss_{}.txt'.format(end)),np.array(loss_list))

    def plot_grad(self,sample_id):
        end=2
        loss=np.loadtxt(join(self.opt_dir,'loss_{}.txt'.format(end)))
        x=np.linspace(0,end,len(loss))
        fig=plt.gcf()
        ax=plt.gca()
        ax.plot(x,loss)
        plot_path=join(self.opt_dir,'loss_{}.png'.format(end))
        print('plot_path',plot_path)
        fig.savefig(plot_path)

    def process(self,out_dir,start,end):
        if not isdir(out_dir):
            os.makedirs(out_dir)
        id_ptr=start
        while id_ptr<=end:
            vts=[]
            sample_ids=[]
            skins=[]
            for sample_id in range(id_ptr,min(id_ptr+self.batch_size,end+1)):
                offset_path=join(self.gt_offset_dir,'offset_{:08d}.npy'.format(sample_id))
                skin_path=join(self.skin_dir,'skin_{:08d}.npy'.format(sample_id))
                if not (isfile(offset_path) and isfile(skin_path)):
                    print(sample_id,'has missing file')
                    continue
                offset=np.load(offset_path)
                skin=np.load(skin_path)
                vt=offset+skin
                # if self.use_lap:
                #     lap=self.L.dot(vt)
                #     vt=np.concatenate([vt,lap],axis=0)
                vt=torch.from_numpy(vt).to(device=self.device,dtype=self.dtype)
                vts.append(vt.unsqueeze(0))
                skins.append(skin)
                sample_ids.append(sample_id)

            vts=torch.cat(vts,dim=0)
            cr_vts=self.proj_module(vts)
            for i in range(len(sample_ids)):
                sample_id=sample_ids[i]
                print('write',sample_id)
                cr_vt=cr_vts[i].cpu().numpy()
                cr_offset=cr_vt-skins[i]
                np.save(join(out_dir,'offset_{:08d}.npy'.format(sample_id)),cr_offset)

            id_ptr+=self.batch_size

    def offset_to_obj(self,offset_dir,sample_id):
        offset=np.load(join(offset_dir,'offset_{:08d}.npy'.format(sample_id)))
        skin=np.load(join(self.skin_dir,'skin_{:08d}.npy'.format(sample_id)))
        vt=offset+skin
        self.write_obj(vt,self.f,join(self.cr_dir,'cloth_1e-2'),sample_id,patch_id=self.patch_id)
        # write_obj(Obj(v=vt,f=self.f),join(self.cr_dir,'test_{:08d}.obj'.format(sample_id)))


if __name__=='__main__':
    # test=CvxOptSimpleTest()
    # test.test()
    test=CvxOptCppTest()
    # test.test(15037)
    # test.test(106)
    # test.test_offset_dir()
    # test.offset_to_obj('/data/zhenglin/poses_v3/lowres_offset_npys',15001)
    # test.offset_to_obj('/data/zhenglin/poses_v3/lowres_offsets_len-2',15001)
    # test.test_samples_dir('../../rundir/lowres_vt_patch/uvn_1e-2/eval_train/waist_front_hip_front_hip_right_waist_right')
    # test.test_samples_dir('../../rundir/lowres_vt/uvn_1e-2/eval_test/')
    # test.avg_samples_dir('../../rundir/lowres_ecos/uvn_1e-2/eval_test')
    # test.test_backward(15037)
    # test.test_loss(106)
    # test.test_grad(15037)
    # test.test_opt(106)
    test.test_plot(15037)
    test.plot_grad(15037)
    # test.test_grad(106)
    # test.process(test.cr_gt_offset_dir,14700,15000)
    # test.offset_to_obj(join(test.data_root_dir,'lowres_offsets_len-2lap1'),107)
    # test.offset_to_obj(join(test.data_root_dir,'lowres_offset_npys'),107)
######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Train_Conv_Mix')
from inequality_opt import InequalitySolver
import torch
import numpy as np
import os
from os.path import join,isdir
from timeit import print_stat,clear_stat
from obj_io import Obj,read_obj,write_obj
from cvxpy_opt_func import init_cvxpy_opt_module,forward_solve,backward_solve
import matplotlib.pyplot as plt

class CvxOptFuncTest:
    def __init__(self):
        self.shared_data_dir='../../shared_data'
        self.data_root_dir='/data/zhenglin/poses_v3'
        # self.gt_dir='opt_test/func_test'
        # self.pd_dir='opt_test/func_test'
        self.cr_dir='opt_test/func_test'
        self.gt_offset_dir=join(self.data_root_dir,'lowres_offset_npys')
        self.skin_dir=join(self.data_root_dir,'lowres_skin_npys')
        self.pd_dir='../../rundir/lowres/xyz/eval_test'
        # self.pd_dir='../../rundir/lowres/xyz/eval_train'
        self.plot_dir='opt_test/func_test/figs'
        if not isdir(self.cr_dir):
            os.makedirs(self.cr_dir)
        if not isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.device=torch.device('cuda:0')
        self.dtype=torch.double

        ctx={'max_num_constraints':-1,'device':self.device,'dtype':self.dtype,'batch_size':2}
        res_ctx={'shared_data_dir':self.shared_data_dir}
        self.cvx_opt_module=init_cvxpy_opt_module(res_ctx,ctx)
        self.opt=self.cvx_opt_module.opts[0]
        self.system=self.cvx_opt_module.system

    def test_forward(self,sample_id):
        # pd_obj=read_obj(join(self.pd_dir,'pd_{:08d}.obj'.format(sample_id)))
        print('forward:{:08d}'.format(sample_id))
        pd_obj=read_obj(join(self.pd_dir,'{:08d}/gt_cloth_{:08d}.obj'.format(sample_id,sample_id)))
        pd_vt=pd_obj.v
        cr_vt,lmd=forward_solve(self.opt,pd_vt)
        print('lmd:min:',np.min(lmd),'max:',np.max(lmd))
        np.save(join(self.cr_dir,'cr_{:08d}.npy'.format(sample_id)),cr_vt)
        np.save(join(self.cr_dir,'lmd_{:08d}.npy'.format(sample_id)),lmd)
        write_obj(Obj(v=cr_vt,f=pd_obj.f),join(self.cr_dir,'cr_gt_{:08d}.obj'.format(sample_id)))

    def test_backward(self,sample_id):
        print('backward:{:08d}'.format(sample_id))
        cr_vt=np.load(join(self.cr_dir,'cr_{:08d}.npy'.format(sample_id)))
        lmd=np.load(join(self.cr_dir,'lmd_{:08d}.npy'.format(sample_id)))
        
        # gt_vt=np.load(join(self.gt_dir,'gt_{:08d}.npy'.format(sample_id)))
        gt_vt=np.load(join(self.gt_offset_dir,'offset_{:08d}.npy'.format(sample_id)))+np.load(join(self.skin_dir,'skin_{:08d}.npy'.format(sample_id)))
        output_grad=cr_vt-gt_vt

        cr_vt=torch.from_numpy(cr_vt).to(device=self.device,dtype=self.dtype)
        lmd=torch.from_numpy(lmd).to(device=self.device,dtype=self.dtype)
        output_grad=torch.from_numpy(output_grad).to(device=self.device,dtype=self.dtype)

        print('cr_vt,min:',torch.min(cr_vt).item(),'max:',torch.max(cr_vt).item())
        print('lmd,min:',torch.min(lmd).item(),'max:',torch.max(lmd).item())
        data=self.system.get_data(cr_vt,lmd)
        f=data['f']
        print('# f>0',torch.sum((f>0).to(dtype=self.dtype)).item())
        print('f,min:',torch.min(f).item(),'max:',torch.max(f).item())
        print('size,f:',f.size(),'lmd:',lmd.size())
        r=lmd/(-f)
        print('# lmd/(-f)<0',torch.sum((r<0).to(dtype=self.dtype)).item())
        print('lmd/(-f),min:',torch.min(r).item(),'max:',torch.max(r).item())
        lmd=lmd.squeeze()
        
        r=torch.log10(r[r>0]+1)
        r=r.cpu().numpy()
        fig=plt.figure()
        ax=plt.gca()
        ax.hist(r,bins=20,log=True)
        ax.set_title('log(1+lmd/(-f))')
        fig.savefig(join(self.plot_dir,'hist_lmd_f.png'))

        grad=backward_solve(cr_vt,output_grad,lmd,self.system,verbose=True)
        print('grad,norm:',torch.norm(grad).item(),'min',torch.min(grad).item(),'max',torch.max(grad).item())

if __name__=='__main__':
    test=CvxOptFuncTest()
    # train_sample_list=np.loadtxt(join(test.data_root_dir,'sample_lists/lowres_train_samples.txt')).astype(int)
    # train_sample_list=train_sample_list[:1]
    train_sample_list=[106]
    for sample_id in train_sample_list:
        test.test_forward(sample_id)
        # test.test_backward(sample_id)

######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import os
from os.path import join,isdir
import cvx_opt_cpp
from cvxpy_opt_func import load_opt_data
from scipy.sparse import csc_matrix
from patch_utils import PatchManager
import time

def write_vts(debug_dir,prefix,vts):
    out_path=join(debug_dir,'{}_{}.npy'.format(prefix,time.strftime('%H:%M:%S',time.localtime(time.time()))))
    np.save(out_path,vts.detach().cpu())

def clean_grad(input_vts,output_vts,out_grad):
    n_samples=len(output_vts)
    for i in range(n_samples):
        if torch.any(torch.isnan(out_grad[i])):
            print('out_grad[{}] is nan, use 0'.format(i))
            out_grad[i]=0
            write_vts('debug','nan_in',input_vts[i])
            write_vts('debug','nan_out',output_vts[i])
        elif torch.any(torch.norm(output_vts[i],dim=1)>6):
            print('output_vts[{}] is unusual, use 0'.format(i))
            out_grad[i]=0
            write_vts('debug','unuaual_in',input_vts[i])
            write_vts('debug','unusual_out',output_vts[i])
    return out_grad

class EcosOptFunction(Function):

    @staticmethod
    def forward(ctx,input_vts,m):
        device,dtype=input_vts.device,input_vts.dtype
        input_vts=input_vts.cpu()
        n_vts=input_vts.size(1)
        if m is None:
            x,y,z,s,success=cvx_opt_cpp.solve_forward(input_vts)
        else:
            m=m.cpu()
            x,y,z,s,success=cvx_opt_cpp.solve_forward_variable_m(input_vts,m)
        output_vts=x[:,:n_vts*3].view(-1,n_vts,3)
        for sample_id in range(len(success)):
            if not success[sample_id]:
                output_vts[sample_id]=input_vts[sample_id] # set to input_vts to avoid gigantic loss
        output_vts=output_vts.to(device=device,dtype=dtype)
        ctx.save_for_backward(x,y,z,s,success,input_vts,output_vts,m)
        return output_vts
        # return x,y,z,s

    @staticmethod
    def backward(ctx,in_grad):
        device,dtype=in_grad.device,in_grad.dtype
        x,y,z,s,success,tgt,cr,m=ctx.saved_tensors
        in_grad=in_grad.cpu()
        if m is None:
            out_grad=cvx_opt_cpp.solve_backward(in_grad,tgt,[x,y,z,s,success])
            out_grad=out_grad.view(out_grad.size(0),-1,3).to(device=device,dtype=dtype)
            return out_grad,None
        else:
            out_grad,out_m_grad=cvx_opt_cpp.solve_backward_variable_m(in_grad,tgt,m,[x,y,z,s,success])
            out_grad=out_grad.view(out_grad.size(0),-1,3).to(device=device,dtype=dtype)
            out_m_grad=out_m_grad.to(device=device,dtype=dtype)
            return out_grad,out_m_grad

ecos_opt=EcosOptFunction.apply

class EcosOptModule(nn.Module):
    def __init__(self):
        super(EcosOptModule,self).__init__()
        
    def forward(self,x,m=None):
        return ecos_opt(x,m)

def load_lap_matrix(shared_data_dir,n_vts):
    L=csc_matrix((n_vts,n_vts))
    Lpr=np.load(join(shared_data_dir,'Lpr.npy'))
    Ljc=np.load(join(shared_data_dir,'Ljc.npy')).astype(np.int)
    Lir=np.load(join(shared_data_dir,'Lir.npy')).astype(np.int)
    return Lpr,Ljc,Lir

def init_ecos_opt_module(res_ctx,ctx):
    print('use_lap:',ctx['use_lap'])
    device,dtype=torch.device('cpu'),ctx['dtype']
    batch_size=ctx['batch_size']
    verbose=ctx['verbose'] if 'verbose' in ctx else False
    use_multi_thread=True
    use_debug=ctx['use_debug']
    cvx_opt_cpp.init(batch_size,verbose,use_multi_thread,use_debug)

    m,edges,l0=load_opt_data(res_ctx,ctx)
    patch_id=ctx['patch_id'] if 'patch_id' in ctx else -1
    if patch_id>=0:
        patch_manager=PatchManager(shared_data_dir=res_ctx['shared_data_dir'])
        patch_vt_ids=patch_manager.load_patch_vt_ids(patch_id)
        patch_edge_ids=patch_manager.get_patch_edge_ids(patch_vt_ids,edges)
        m=m[patch_vt_ids]
        l0=l0[patch_edge_ids]
        edges=patch_manager.get_patch_edges(patch_id,edges)
    m=torch.from_numpy(m).to(device=device,dtype=dtype)
    edges=torch.from_numpy(edges).to(device=device,dtype=torch.long)
    l0=torch.from_numpy(l0).to(device=device,dtype=dtype)
    if ctx['use_spring']:
        k=res_ctx['stiffness'] # set by load_opt_data
        if patch_id>=0:
            k=k[patch_edge_ids]
        res_ctx['stiffness']=k
        k=torch.from_numpy(k).to(device=device,dtype=dtype)
        cvx_opt_cpp.init_spring(k,ctx['lmd_k'])
        print('init_spring')

    if ctx['use_lap']:
        assert(patch_id<0)
        Lpr_np,Ljc_np,Lir_np=load_lap_matrix(res_ctx['shared_data_dir'],len(m))
        Lpr=torch.from_numpy(Lpr_np).to(device=device,dtype=dtype)
        Ljc=torch.from_numpy(Ljc_np[1:]).to(device=device,dtype=torch.long)
        Lir=torch.from_numpy(Lir_np).to(device=device,dtype=torch.long)
        print('init_lap')
        cvx_opt_cpp.init_lap(Lpr,Ljc,Lir,ctx['lmd_lap'])

    cvx_opt_cpp.init_forward(m,edges,l0)
    cvx_opt_cpp.init_backward()
    print('n_edges:',len(edges))

    maxit=ctx.get('maxit',50)
    feastol=ctx.get('feastol',1e-6)
    abstol=ctx.get('abstol',1e-6)
    reltol=ctx.get('reltol',5e-1)
    # reltol=ctx.get('reltol',1e-6)
    feastol_inacc=ctx.get('feastol_inacc',5e-3)
    abstol_inacc=ctx.get('abstol_inacc',1e-2)
    reltol_inacc=ctx.get('reltol_inacc',5e0)
    print('maxit',maxit,'feastol',feastol,'abstol',abstol,'reltol',reltol,'feastol_inacc',feastol_inacc,'abstol_inacc',abstol_inacc,'reltol_inacc',reltol_inacc)
    cvx_opt_cpp.init_options(maxit,feastol,abstol,reltol,feastol_inacc,abstol_inacc,reltol_inacc)

    module=EcosOptModule()
    module.edges=edges.numpy()
    module.m=m.numpy()
    module.l0=l0.numpy()
    if ctx['use_spring']:
        module.k=k

    if ctx['use_lap']:
        n_vts=len(m)
        module.L=csc_matrix((Lpr_np,Lir_np,Ljc_np),(n_vts,n_vts))

    return module
######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
from timeit import timestat
from cg_solve import cg_solve
class InequalityOptSystem:
    def __init__(self,m,l02,Is):
        self.m=m
        self.l02=l02
        self.i0,self.i1,self.I0,self.I1=Is
        self.use_jacobi_precondition=False
        self.use_vt_jocabi_precondition=True

    def get_Hu(self,data,u):
        f,df,lmd=data['f'],data['df'],data['lmd']
        du=u[self.i1]-u[self.i0]
        tu=lmd*(df/(-f)*torch.sum(df*du,dim=1,keepdim=True)+du)
        Hu=self.m*u
        Hu.scatter_add_(0,self.I0,-tu)
        Hu.scatter_add_(0,self.I1,tu)
        if torch.any(torch.isnan(u)):
            assert(False)
        return Hu

    def get_b(self,data,x,t):
        x0,f=data['x0'],data['f']
        b=self.m*(x0-x)
        ldx=(x[self.i1]-x[self.i0])/f/t
        b.scatter_add_(0,self.I0,-ldx)
        b.scatter_add_(0,self.I1,ldx)
        return b

    def mul(self,data,u):
        return self.get_Hu(data,u)

    def inner(self,x,y):
        return torch.sum(x*y)

    def convergence_norm(self,r):
        return torch.sqrt(torch.max(torch.sum(r**2,dim=1)))  

    def get_df(self,x):
        return x[self.i1]-x[self.i0]

    def get_f(self,df):
        return (torch.sum(df**2,dim=1,keepdim=True)-self.l02)/2

    def get_data(self,x,lmd):
        data={'size':x.size(),'device':x.device,'dtype':x.dtype}
        data['df']=self.get_df(x)
        data['f']=self.get_f(data['df'])
        data['lmd']=lmd
        return data

    def jacobi_precondition(self,data,r):
        f,df,lmd=data['f'],data['df'],data['lmd']
        tm=lmd/(-f)*(df*df)+lmd
        m=self.m.repeat(1,3)
        m.scatter_add_(0,self.I0,tm)
        m.scatter_add_(0,self.I1,tm)
        return r/m

    def vt_jacobi_precondition(self,data,r):
        f,df,lmd=data['f'],data['df'],data['lmd']
        tm=torch.bmm(df.view(-1,3,1),df.view(-1,1,3))*(lmd/(-f)).view(-1,1,1)
        tm=tm.view(-1,9)
        tm[:,[0,4,8]]+=lmd
        m=torch.zeros((len(self.m),9),device=data['device'],dtype=data['dtype'])
        m[:,[0,4,8]]=self.m
        m.scatter_add_(0,self.I0.repeat(1,3),tm)
        m.scatter_add_(0,self.I1.repeat(1,3),tm)
        m=torch.inverse(m.view(-1,3,3))
        return torch.matmul(m,r.view(-1,3,1)).view(-1,3)

    def precondition(self,data,r):
        if self.use_jacobi_precondition:
            return self.jacobi_precondition(data,r)
        elif self.use_vt_jocabi_precondition:
            return self.vt_jacobi_precondition(data,r)
        else:
            return r

    def share_memory(self):
        self.m.share_memory_()
        self.l02.share_memory_()
        self.i0.share_memory_()
        self.i1.share_memory_()
        self.I0.share_memory_()
        self.I1.share_memory_()


class InequalitySolver:
    def __init__(self,m,edges,l0):
        self.m=m
        self.edges=edges
        self.l0=l0
        self.l02=l0**2
        self.i0=edges[:,0]
        self.i1=edges[:,1]
        self.I0=edges[:,:1].repeat(1,3)
        self.I1=edges[:,1:2].repeat(1,3)
        self.system=InequalityOptSystem(m,self.l02,(self.i0,self.i1,self.I0,self.I1))
        self.n_ineq=len(edges)
        self.mu=10
        self.cg_tol=1e-5
        self.cg_max_iter=len(m)*3
        self.tol_dual=2e-4
        self.tol_eta=1e-4
        self.beta=0.5
        self.alpha=0.01
        self.max_ls_trials=64
        self.verbose=True

    @timestat
    def get_dx(self,data,x,lmd,x0,t):
        data['x0']=x0
        dx0=torch.zeros_like(x,device=x.device,dtype=x.dtype)
        b=self.system.get_b(data,x,t)
        dx,cg_iters=cg_solve(self.system,data,dx0,b,tol=self.cg_tol,max_iterations=self.cg_max_iter)
        return dx,cg_iters

    @timestat
    def get_dlmd(self,data,dx,lmd,t):
        f,df=data['f'],data['df']
        ddx=dx[self.i1]-dx[self.i0]
        dlmd=(-lmd*torch.sum(ddx*df,dim=1,keepdim=True)-1/t)/f-lmd
        return dlmd

    @timestat
    def line_search(self,x,lmd,dx,dlmd,x0,t,max_s=1):
        def get_r2(sx,slmd):
            return torch.sum(self.get_r_dual(sx,x0,slmd)**2)+torch.sum(self.get_r_cent(sx,slmd,t)**2)
        s=max_s
        r=get_r2(x,lmd)
        for i in range(self.max_ls_trials):
            sx=x+dx*s
            slmd=lmd+dlmd*s
            sr=get_r2(sx,slmd)
            if sr<=(1-self.alpha*s)**2*r:
                return sx,slmd,s
            s*=self.beta
        print('not a descending direction')
        assert(False)

    def get_eta(self,f,lmd):
        return -torch.sum(f*lmd)

    @timestat
    def get_max_s(self,x,dx,lmd,dlmd):
        s=1
        slmd=lmd+s*dlmd
        lmd_ids=slmd<0
        if torch.any(lmd_ids):
            s=torch.min(-lmd[lmd_ids]/dlmd[lmd_ids])
        s*=0.99
        for i in range(self.max_ls_trials):
            sx=x+s*dx
            sf=self.system.get_f(self.system.get_df(sx))
            if torch.all(sf<0): # hack
                return s
            s*=self.beta
        print('no feasible s')
        assert(False)

    def get_r_dual(self,x,x0,lmd):
        du=(x[self.i1]-x[self.i0])*lmd
        r_dual=(x-x0)*self.m
        r_dual.scatter_add_(0,self.I0,-du)
        r_dual.scatter_add_(0,self.I1,du)
        return r_dual

    def get_r_cent(self,x,lmd,t):
        df=self.system.get_df(x)
        f=self.system.get_f(df)
        return -f*lmd-1/t

    def converged(self,r_dual_norm,eta):
        return r_dual_norm<self.tol_dual and eta<self.tol_eta

    @timestat
    def solve(self,x0):
        x=torch.zeros_like(x0,device=x0.device,dtype=x0.dtype)
        # lmd=2/(self.l02)*100
        lmd=torch.ones((len(self.l02),1),device=x0.device,dtype=x0.dtype)
        max_iter=10000
        for i in range(max_iter):
            data=self.get_data(x,lmd)
            eta=self.get_eta(data['f'],lmd)
            r_dual=self.get_r_dual(x,x0,lmd)
            r_dual_norm=torch.norm(r_dual)
            if self.converged(r_dual_norm,eta):
                return x
            t=self.mu*self.n_ineq/eta
            dx,cg_iters=self.get_dx(data,x,lmd,x0,t)
            dlmd=self.get_dlmd(data,dx,lmd,t)
            max_s=self.get_max_s(x,dx,lmd,dlmd)
            x,lmd,s=self.line_search(x,lmd,dx,dlmd,x0,t,max_s=max_s)
            if self.verbose:
                print('iter:{:04d},t:{:.2E},eta:{:.8f},r_dual:{:.8f},max_s:{:.8f},s:{:.8f},cg_iters:{}'.format(i,t.item(),eta.item(),r_dual_norm.item(),max_s,s,cg_iters))
        print('exceed max_iter',max_iter)
        assert(False)



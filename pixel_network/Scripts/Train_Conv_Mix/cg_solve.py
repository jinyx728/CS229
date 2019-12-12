######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch 
import numpy as np

def cg_solve(system,A,x,b,tol=1e-8,max_iterations=10000):
    rho_old=0
    convergence_norm=0
    iterations=0
    FLT_MAX=1e6
    while True:
        restart=iterations==0
        if restart:
            r=b-system.mul(A,x)
        convergence_norm=system.convergence_norm(A,r)
        # print('iter',iterations,'convergence_norm:',convergence_norm.item())
        if convergence_norm<=tol:
            return x,iterations
        if iterations==max_iterations:
            return x,-1

        iterations+=1
        mr=system.precondition(A,r)
        rho=system.inner(mr,r)
        if restart:
            s=mr
        else:
            s=rho/rho_old*s+mr
        q=system.mul(A,s)
        s_dot_q=system.inner(s,q)
        if s_dot_q<=0:
            print('CG appears to be indefinite or singular, s_dot_q/s_dot_s=',(s_dot_q/system.inner(s,s)).item())
            assert(False)
        if s_dot_q>0:
            alpha=rho/s_dot_q
        else:
            alpha=FLT_MAX
        x+=alpha*s
        r-=alpha*q
        rho_old=rho

 

if __name__=='__main__':
    class TestSystem:
        def mul(self,A,x):
            return torch.matmul(A,x)

        def inner(self,x,y):
            return torch.sum(x*y)

        def convergence_norm(self,r):
            return torch.sqrt(torch.max(torch.sum(r**2)))    

    system=TestSystem()
    A=torch.tensor([[2,0,-1],
                    [0,1,0],
                    [-1,0,3]],dtype=torch.double)
    b=torch.tensor([1,1,2],dtype=torch.double)
    x0=torch.zeros(3,dtype=torch.double)
    print(cg_solve(system,A,x0,b))

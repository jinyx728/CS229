######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import cvx_opt_cpp

def test():
	x=torch.tensor([[2,0,0,0,0,0]],dtype=torch.double)
	cvx_opt_cpp.solve(x)

if __name__=='__main__':
	test()
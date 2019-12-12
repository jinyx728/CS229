######################################################################
# Copyright 2019. Dan Johnson.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch

print("torch.cuda.is_available() = " + str(torch.cuda.is_available()))
cur_device = 0
print(torch.cuda.get_device_name(cur_device))
device = torch.device("cuda:"+str(cur_device))
x = torch.rand((2, 2), device=device)
print(x)

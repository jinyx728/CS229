######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

def cat_coord(m):
    n_samples,Ny,Nx=m.size(0),m.size(2),m.size(3)
    x=torch.linspace(-1+1/(2*Nx),1-1/(2*Nx),steps=Nx,device=m.device,dtype=m.dtype).view(1,1,1,-1).repeat(n_samples,1,Ny,1)
    y=torch.linspace(-1+1/(2*Ny),1-1/(2*Ny),steps=Ny,device=m.device,dtype=m.dtype).view(1,1,-1,1).repeat(n_samples,1,1,Nx)
    return torch.cat([m,x,y],dim=1)

class R(nn.Module):
    def __init__(self,input_channels=6,input_size=256,output_size=90,use_coord_conv=True):
        super(self.__class__,self).__init__()
        channels=[128,128,128,128,128,128,128]
        self.use_coord_conv=use_coord_conv
        if use_coord_conv:
            input_channels+=2
        modules=[nn.Conv2d(input_channels,channels[0],4,2,1),
                 nn.BatchNorm2d(channels[0]),
                 nn.LeakyReLU(0.2)]
        block_id=1
        size=input_size//2
        while size>4:
            modules=modules+[
                nn.Conv2d(channels[block_id-1],channels[block_id],4,2,1),
                nn.BatchNorm2d(channels[block_id]),
                nn.LeakyReLU(0.2)]
            block_id+=1
            size//=2

        modules.append(nn.Conv2d(channels[block_id],output_size,size,1,0))
        self.main=nn.Sequential(*modules)

    def forward(self,x):
        if self.use_coord_conv:
            x=cat_coord(x)
        return self.main(x)


######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from progressive_gan_utils import Reshape,Upsample


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.LeakyReLU(0.2)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.tol=1e-3

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class G(nn.Module):
    def __init__(self,input_size=90,output_size=256,output_channels=6,init_size=8):
        super(self.__class__,self).__init__()
        channels=[64,64,32,16,8,output_channels]
        self.init_block=nn.Sequential(
            nn.Linear(input_size,init_size*init_size*channels[0]),
            Reshape((-1,channels[0],init_size,init_size)),
            norm(channels[0]),
            nn.LeakyReLU(0.2)
        )
        module_list=[]
        block_id=0
        size=init_size
        while size<output_size:
            module_list.append(nn.Sequential(
                nn.ConvTranspose2d(channels[block_id],channels[block_id+1],4,2,1),
                norm(channels[block_id+1]),
                ODEBlock(ODEfunc(channels[block_id+1])),
            ))
            block_id+=1
            size*=2
        self.module_list=nn.Sequential(*module_list)

    def forward(self,x):
        return self.module_list(self.init_block(x))


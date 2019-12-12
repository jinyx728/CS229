######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import trace

class Conv2d(nn.Module):
    def __init__(self,kernel_size,stride,padding,in_channels,out_channels,gain=1):
        super(Conv2d,self).__init__()
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.in_channels=in_channels
        self.out_channels=out_channels

        self.weight=nn.Parameter(torch.Tensor(out_channels,in_channels,kernel_size,kernel_size))
        self.bias=nn.Parameter(torch.Tensor(out_channels))
        nn.init.normal_(self.weight.data)
        self.bias.data.fill_(0)

        nl=self.kernel_size*self.kernel_size*self.in_channels
        self.c=torch.sqrt(torch.tensor(2/nl))*gain

    def forward(self,x):
        y=F.conv2d(x,self.weight*self.c,bias=self.bias,stride=self.stride,padding=self.padding)
        return y

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}')
        return s.format(**self.__dict__)

def cat_coord(m):
    n_samples,Ny,Nx=m.size(0),m.size(2),m.size(3)
    x=torch.linspace(-1+1/(2*Nx),1-1/(2*Nx),steps=Nx,device=m.device,dtype=m.dtype).view(1,1,1,-1).repeat(n_samples,1,Ny,1)
    y=torch.linspace(-1+1/(2*Ny),1-1/(2*Ny),steps=Ny,device=m.device,dtype=m.dtype).view(1,1,-1,1).repeat(n_samples,1,1,Nx)
    return torch.cat([m,x,y],dim=1)

class CoordConv2d(Conv2d):
    def __init__(self,kernel_size,stride,padding,in_channels,out_channels,gain=1):
        super(CoordConv2d,self).__init__(kernel_size,stride,padding,in_channels+2,out_channels,gain)

    def forward(self,x):
        x=cat_coord(x)
        return super(self.__class__,self).forward(x)

class Linear(nn.Module):
    def __init__(self,in_features,out_features,gain=1):
        super(self.__class__,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.gain=gain
        self.weight=nn.Parameter(torch.Tensor(out_features,in_features))
        self.bias=nn.Parameter(torch.Tensor(out_features))
        nn.init.normal_(self.weight.data)
        self.bias.data.fill_(0)

        nl=self.in_features
        self.c=torch.sqrt(torch.tensor(2/nl))*self.gain

    def forward(self,x):
        return F.linear(x,self.weight*self.c,self.bias)

    def extra_repr(self):
        s = ('{in_features}, {out_features}')
        return s.format(**self.__dict__)

class LeakyReLU(nn.Module):
    def __init__(self):
        super(self.__class__,self).__init__()

    def forward(self,x):
        return F.leaky_relu(x,negative_slope=0.2)

class Reshape(nn.Module):
    def __init__(self,out_shape):
        super(self.__class__,self).__init__()
        self.out_shape=out_shape
    def forward(self,x):
        return x.view(self.out_shape)

class Upsample(nn.Module):
    def __init__(self):
        super(self.__class__,self).__init__()

    def forward(self,x):
        return F.interpolate(x,scale_factor=2,mode='bilinear')

class Downsample(nn.Module):
    def __init__(self):
        super(self.__class__,self).__init__()

    def forward(self,x):
        return F.interpolate(x,scale_factor=1/2,mode='bilinear')

class LayerResponseNorm2d(nn.Module):
    def __init__(self):
        super(self.__class__,self).__init__()

    def forward(self,x):
        eps=1e-8
        norm=torch.mean(x**2,dim=1,keepdim=True)
        norm[norm<eps]=eps
        norm=torch.sqrt(norm)
        return x/norm

class BatchStd(nn.Module):
    def __init__(self):
        super(self.__class__,self).__init__()

    def forward(self,x):
        N,D,H,W=x.size()
        eps=1e-8
        var=torch.mean(x**2,dim=0)-torch.mean(x,dim=0)**2
        var[var<eps]=eps
        std=torch.sqrt(var)
        std=torch.mean(std[std>=2e-4]) # hack
        # std=torch.mean(torch.std(x,dim=0)) produce nan in backward process 
        return torch.cat([x,torch.ones((N,1,H,W),device=x.device,dtype=x.dtype)*std],dim=1) # trace will memorize this device here. Tricky

def forward(module_list,x):
    for module in module_list:
        x=module(x)
    return x

class G(nn.Module):
    def __init__(self,start_level=3,end_level=8,input_size=90,output_channels=6,channels=None):
        super(self.__class__,self).__init__()
        assert(channels is not None)
        init_size=2**start_level
        self.start_level=start_level
        self.end_level=end_level
        self.input_size=input_size
        self.channels=channels

        self.init_block=nn.Sequential(
            Linear(input_size,init_size*init_size*channels[start_level],gain=1),
            Reshape((-1,channels[start_level],init_size,init_size)),
            # LayerResponseNorm2d(),
            nn.InstanceNorm2d(channels[start_level]),
            # nn.InstanceNorm2d(channels[start_level]),
            LeakyReLU(),
            Conv2d(3,1,1,channels[start_level],channels[start_level]),
            # LayerResponseNorm2d(),
            nn.InstanceNorm2d(channels[start_level]),
            # nn.InstanceNorm2d(channels[start_level]),
            LeakyReLU()
            )

        # input size to level_blocks[level] is 2^level
        self.level_blocks=[None for i in range(end_level)]
        for level in range(start_level,end_level):
            self.level_blocks[level]=nn.Sequential(
                Upsample(),
                Conv2d(3,1,1,channels[level],channels[level+1]),
                # LayerResponseNorm2d(),
                nn.InstanceNorm2d(channels[level+1]),
                # nn.InstanceNorm2d(channels[level+1]),
                LeakyReLU(),
                Conv2d(3,1,1,channels[level+1],channels[level+1]),
                # LayerResponseNorm2d(),
                nn.InstanceNorm2d(channels[level+1]),
                # nn.InstanceNorm2d(channels[level+1]),
                LeakyReLU())
        self.level_blocks=nn.ModuleList(self.level_blocks)

        self.toRGBs=[None for i in range(end_level+1)]
        for level in range(start_level,end_level+1):
            self.toRGBs[level]=Conv2d(1,1,0,channels[level],output_channels)
        self.toRGBs=nn.ModuleList(self.toRGBs)

    def trace(self):
        self.init_block=trace(self.init_block,torch.rand(1,self.input_size))
        for level in range(self.start_level,self.end_level):
            self.level_blocks[level]=trace(self.level_blocks[level],torch.rand(1,self.channels[level],2**level,2**level))
        for level in range(self.start_level,self.end_level+1):
            self.toRGBs[level]=trace(self.toRGBs[level],torch.rand(1,self.channels[level],2**level,2**level))

    def forward(self,x,out_level,alpha):
        x=self.init_block(x)
        if out_level==self.start_level:
            return self.toRGBs[out_level](x) # no blending
        for level in range(self.start_level,out_level-1):
            x=self.level_blocks[level](x)
        if alpha<1 and alpha>0:
            prev_rgb=self.toRGBs[out_level-1](x)
        x=self.level_blocks[out_level-1](x)
        rgb=self.toRGBs[out_level](x)
        if alpha<1 and alpha>0:
            rgb=alpha*rgb+(1-alpha)*F.interpolate(prev_rgb,scale_factor=2)
        return rgb

class D(nn.Module):
    def __init__(self,start_level=3,end_level=8,input_channels=6,channels=None,output_size=1):
        super(self.__class__,self).__init__()
        assert(channels is not None)
        init_size=2**start_level
        self.start_level=start_level
        self.end_level=end_level
        self.channels=channels
        self.input_channels=input_channels
        # output size of level_blocks[level] is 2^level
        self.level_blocks=[None for i in range(end_level)]
        for level in range(start_level,end_level):
            self.level_blocks[level]=nn.Sequential(
                Conv2d(3,1,1,channels[level+1],channels[level]),
                # LayerResponseNorm2d(),
                nn.InstanceNorm2d(channels[level]),
                # nn.BatchNorm2d(channels[level]),
                LeakyReLU(),
                Conv2d(3,1,1,channels[level],channels[level]),
                # LayerResponseNorm2d(),
                nn.InstanceNorm2d(channels[level]),
                # nn.BatchNorm2d(channels[level]),
                LeakyReLU(),
                # Downsample() # not working
                nn.AvgPool2d(kernel_size=2)
                )
        self.level_blocks=nn.ModuleList(self.level_blocks)
        self.fromRGBs=[None for i in range(end_level+1)]
        for level in range(start_level,end_level+1):
            self.fromRGBs[level]=Conv2d(1,1,0,input_channels,channels[level])
        self.fromRGBs=nn.ModuleList(self.fromRGBs)

        self.final_block=nn.Sequential(
            BatchStd(),
            Conv2d(3,1,1,channels[start_level]+1,channels[start_level]),
            # LayerResponseNorm2d(),
            nn.InstanceNorm2d(channels[start_level]),
            # nn.BatchNorm2d(channels[start_level]),
            LeakyReLU(),
            Conv2d(init_size,1,0,channels[start_level],channels[start_level]),
            # LayerResponseNorm2d(),
            # nn.InstanceNorm2d(channels[start_level]),
            # nn.BatchNorm2d(channels[start_level]),
            LeakyReLU(),
            Reshape((-1,1,1,channels[start_level])),
            Linear(channels[start_level],output_size,gain=torch.sqrt(torch.tensor(1/2)))
            )
        # self.trace()

    def trace(self,device):
        for level in range(self.start_level,self.end_level+1):
            self.fromRGBs[level]=trace(self.fromRGBs[level],torch.rand(1,self.input_channels,2**level,2**level,device=device))
        for level in range(self.start_level,self.end_level):
            self.level_blocks[level]=trace(self.level_blocks[level],torch.rand(1,self.channels[level+1],2**(level+1),2**(level+1),device=device))
        self.final_block=trace(self.final_block,torch.rand(1,self.channels[self.start_level],2**self.start_level,2**self.start_level,device=device))


    def forward(self,x,in_level,alpha):
        rgb=x
        x=self.fromRGBs[in_level](x)
        if in_level==self.start_level:
            return self.final_block(x)
        x=self.level_blocks[in_level-1](x)
        if alpha<1 and alpha>0:
            prev_x=self.fromRGBs[in_level-1](F.interpolate(rgb,scale_factor=0.5))
            x=alpha*x+(1-alpha)*prev_x
        for level in reversed(range(self.start_level,in_level-1)):
            x=self.level_blocks[level](x)
        return self.final_block(x)







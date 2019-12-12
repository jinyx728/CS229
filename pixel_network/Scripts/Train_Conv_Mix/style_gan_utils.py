######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from progressive_gan_utils import Conv2d,CoordConv2d,Linear,LeakyReLU,Reshape,Upsample,Downsample,BatchStd

class AdpIN2d(nn.Module):
    def __init__(self,in_channels):
        super(self.__class__,self).__init__()
        self.norm=nn.InstanceNorm2d(in_channels,affine=False,track_running_stats=False)

    def forward(self,x,gamma,beta):
        gamma=gamma.view(gamma.size(0),gamma.size(1),1,1)
        beta=beta.view(beta.size(0),beta.size(1),1,1)
        return self.norm(x)*gamma+beta

class InstanceNorm1d(nn.Module):
    def __init__(self,in_channels):
        super(self.__class__,self).__init__()
        self.norm=nn.InstanceNorm1d(in_channels)
    def forward(self,x):
        x=x.view(x.size(0),1,-1)
        return self.norm(x).squeeze()

class UpLevelBlock(nn.Module):
    def __init__(self,in_channels,out_channels,use_coord_conv=True):
        super(self.__class__,self).__init__()
        self.out_channels=out_channels
        if use_coord_conv:
            conv=CoordConv2d
        else:
            conv=Conv2d
        self.module_list=nn.ModuleList([
            Upsample(),
            conv(3,1,1,in_channels,out_channels),
            AdpIN2d(out_channels),
            LeakyReLU(),
            conv(3,1,1,out_channels,out_channels),
            AdpIN2d(out_channels),
            LeakyReLU()])
    def forward(self,x,coeffs):
        nc=self.out_channels
        for i in range(len(self.module_list)):
            module=self.module_list[i]
            if i==2:
                gamma,beta=coeffs[:,:nc]+1,coeffs[:,nc:2*nc]
                x=module(x,gamma,beta)
            elif i==5:
                gamma,beta=coeffs[:,2*nc:3*nc]+1,coeffs[:,3*nc:4*nc]
                x=module(x,gamma,beta)
            else:
                x=module(x)
        return x

class InitBlock(nn.Module):
    def __init__(self,input_size,init_size,init_channels,use_coord_conv=False):
        super(self.__class__,self).__init__()
        self.out_channels=init_channels
        if use_coord_conv:
            conv=CoordConv2d
        else:
            conv=Conv2d
        self.module_list=nn.ModuleList([
            Linear(input_size,init_size*init_size*init_channels,gain=1),
            Reshape((-1,init_channels,init_size,init_size)),
            AdpIN2d(init_channels),
            LeakyReLU(),
            conv(3,1,1,init_channels,init_channels),
            AdpIN2d(init_channels),
            LeakyReLU()])
    def forward(self,x,coeffs):
        nc=self.out_channels
        for i in range(len(self.module_list)):
            module=self.module_list[i]
            if i==2:
                gamma,beta=coeffs[:,:nc]+1,coeffs[:,nc:2*nc]
                x=module(x,gamma,beta)
            elif i==5:
                gamma,beta=coeffs[:,2*nc:3*nc]+1,coeffs[:,3*nc:4*nc]
                x=module(x,gamma,beta)
            else:
                x=module(x)
        return x

def get_feat_block(feat_channels):
    feat_block=[]
    for i in range(len(feat_channels)-1):
        feat_block+=[Linear(feat_channels[i],feat_channels[i+1]),
                        InstanceNorm1d(feat_channels[i+1]),
                        LeakyReLU()]
    return nn.Sequential(*feat_block)

def get_label(label,x):
    if label=='hero':
        return torch.tensor([[1,0]],device=x.device,dtype=x.dtype).repeat(x.size(0),1)
    elif label=='sim':
        return torch.tensor([[0,1]],device=x.device,dtype=x.dtype).repeat(x.size(0),1)
    else:
        assert(False)

class G(nn.Module):
    def __init__(self,start_level=3,end_level=8,input_size=90,output_channels=6,channels=None,use_coord_conv=True,ctx=None):
        super(self.__class__,self).__init__()
        assert(channels is not None)
        init_size=2**start_level

        self.start_level=start_level
        self.end_level=end_level
        self.input_size=input_size
        self.channels=channels
        self.use_label=ctx['use_label']
        self.use_separate_block=ctx['use_separate_block']

        if use_coord_conv: # only use conv in level blocks to save parameters and space
            conv=CoordConv2d
        else:
            conv=Conv2d

        feat_size=128
        if self.use_label:
            if self.use_separate_block:
                self.init_block=nn.ModuleList([InitBlock(input_size,init_size,channels[start_level]),InitBlock(input_size,init_size,channels[start_level])]) 
            else:
                self.init_block=InitBlock(input_size,init_size,channels[start_level])
            self.cond_feat_block=get_feat_block([input_size,feat_size])
            self.label_feat_block=get_feat_block([2,64])
            self.merge_feat_block=get_feat_block([feat_size+64,feat_size+32,feat_size])
        else:
            self.init_block=InitBlock(input_size,init_size,channels[start_level])
            self.cond_feat_block=get_feat_block([input_size,feat_size,feat_size])
            # self.style_feat_block=get_feat_block([input_size,feat_size,feat_size])

        self.init_style_block=Linear(feat_size,4*channels[start_level])

        level_style_blocks=[None for i in range(end_level)]
        for level in range(start_level,end_level):
            level_style_blocks[level]=Linear(feat_size,4*channels[level+1])
        self.level_style_blocks=nn.ModuleList(level_style_blocks)

        # input size to level_blocks[level] is 2^level
        self.level_blocks=[None for i in range(end_level)]
        for level in range(start_level,end_level):
            self.level_blocks[level]=UpLevelBlock(channels[level],channels[level+1],use_coord_conv=use_coord_conv)
        self.level_blocks=nn.ModuleList(self.level_blocks)

        self.toRGBs=[None for i in range(end_level+1)]
        for level in range(start_level,end_level+1):
            self.toRGBs[level]=Conv2d(1,1,0,channels[level],output_channels)
        self.toRGBs=nn.ModuleList(self.toRGBs)

    def forward(self,x,out_level,alpha,label=None):
        cond=x
        style_feat=self.cond_feat_block(cond)
        # style_feat=self.style_feat_block(cond)
        if self.use_label:
            label_feat=self.label_feat_block(get_label(label,x))
            style_feat=torch.cat([style_feat,label_feat],dim=1)
            style_feat=self.merge_feat_block(style_feat)
            init_style=self.init_style_block(style_feat)
            block_id=0 if label=='hero' else 1
            if self.use_separate_block:
                x=self.init_block[block_id](x,init_style)
            else:
                x=self.init_block(x,init_style)
        else:
            init_style=self.init_style_block(style_feat)
            x=self.init_block(x,init_style)
        if out_level==self.start_level:
            return self.toRGBs[out_level](x) # no blending
        for level in range(self.start_level,out_level-1):
            level_style=self.level_style_blocks[level](style_feat)
            x=self.level_blocks[level](x,level_style)
        if alpha<1 and alpha>0:
            prev_rgb=self.toRGBs[out_level-1](x)
        level_style=self.level_style_blocks[out_level-1](style_feat)
        x=self.level_blocks[out_level-1](x,level_style)
        rgb=self.toRGBs[out_level](x)
        if alpha<1 and alpha>0:
            rgb=alpha*rgb+(1-alpha)*F.interpolate(prev_rgb,scale_factor=2)
        return rgb

class AdpIN1d(nn.Module):
    def __init__(self,in_channels):
        super(self.__class__,self).__init__()
        self.norm=InstanceNorm1d(in_channels)

    def forward(self,x,gamma,beta):
        return self.norm(x)*(gamma+1)+beta

class DownLevelBlock(nn.Module):
    def __init__(self,in_channels,out_channels,use_coord_conv=False):
        super(self.__class__,self).__init__()
        self.out_channels=out_channels
        if use_coord_conv:
            conv=CoordConv2d
        else:
            conv=Conv2d
        self.module_list=nn.ModuleList([
            Conv2d(3,1,1,in_channels,out_channels),
            AdpIN2d(out_channels),
            LeakyReLU(),
            Conv2d(3,1,1,out_channels,out_channels),
            AdpIN2d(out_channels),
            LeakyReLU(),
            nn.AvgPool2d(kernel_size=2)])

    def forward(self,x,coeffs):
        nc=self.out_channels
        for i in range(len(self.module_list)):
            module=self.module_list[i]
            if i==1:
                gamma,beta=coeffs[:,:nc],coeffs[:,nc:2*nc]
                x=module(x,gamma,beta)
            elif i==4:
                gamma,beta=coeffs[:,2*nc:3*nc],coeffs[:,3*nc:4*nc]
                x=module(x,gamma,beta)
            else:
                x=module(x)
        return x

class FinalBlock(nn.Module):
    def __init__(self,in_channels,init_size,use_coord_conv=False):
        super(self.__class__,self).__init__()
        self.out_channels=in_channels
        if use_coord_conv:
            conv=CoordConv2d
        else:
            conv=Conv2d
        self.module_list=nn.ModuleList([
            BatchStd(),
            Conv2d(3,1,1,in_channels+1,in_channels),
            AdpIN2d(in_channels),
            LeakyReLU(),
            Conv2d(init_size,1,0,in_channels,in_channels),
            Reshape((-1,in_channels)),
            AdpIN1d(in_channels),
            LeakyReLU(),
            Linear(in_channels,1)])

    def forward(self,x,coeffs):
        nc=self.out_channels
        for i in range(len(self.module_list)):
            module=self.module_list[i]
            if i==2:
                gamma,beta=coeffs[:,:nc]+1,coeffs[:,nc:2*nc]
                x=module(x,gamma,beta)
            elif i==6:
                gamma,beta=coeffs[:,2*nc:3*nc]+1,coeffs[:,3*nc:4*nc]
                x=module(x,gamma,beta)
            else:
                x=module(x)
        return x

class D(nn.Module):
    def __init__(self,start_level=3,end_level=8,input_channels=6,channels=None,cond_size=90,output_size=1,use_coord_conv=True,ctx=None):
        super(self.__class__,self).__init__()
        assert(channels is not None)
        init_size=2**start_level
        self.start_level=start_level
        self.end_level=end_level
        self.channels=channels
        self.input_channels=input_channels
        self.use_label=ctx['use_label']
        self.use_separate_block=ctx['use_separate_block']

        if use_coord_conv: # only use conv in fromRGBs to save parameters and space
            conv=CoordConv2d
        else:
            conv=Conv2d

        feat_size=128

        if self.use_label:
            self.style_feat_block=get_feat_block([cond_size,feat_size])
            self.label_feat_block=get_feat_block([2,64])
            self.merge_feat_block=get_feat_block([feat_size+64,feat_size+32,feat_size])
            self.final_style_block=Linear(feat_size,4*channels[start_level])
            if self.use_separate_block:
                self.final_block=nn.ModuleList([FinalBlock(channels[start_level],init_size),FinalBlock(channels[start_level],init_size)])
            else:
                self.final_block=FinalBlock(channels[start_level],init_size)
        else:
            self.style_feat_block=get_feat_block([cond_size,feat_size,feat_size])
            self.final_style_block=Linear(feat_size,4*channels[start_level])
            self.final_block=FinalBlock(channels[start_level],init_size)

        level_style_blocks=[None for i in range(end_level)]
        for level in range(start_level,end_level):
            level_style_blocks[level]=Linear(feat_size,4*channels[level])
        self.level_style_blocks=nn.ModuleList(level_style_blocks)

        # output size of level_blocks[level] is 2^level
        self.level_blocks=[None for i in range(end_level)]
        for level in range(start_level,end_level):
            self.level_blocks[level]=DownLevelBlock(channels[level+1],channels[level])
        self.level_blocks=nn.ModuleList(self.level_blocks)
        self.fromRGBs=[None for i in range(end_level+1)]
        for level in range(start_level,end_level+1):
            self.fromRGBs[level]=conv(1,1,0,input_channels,channels[level])
        self.fromRGBs=nn.ModuleList(self.fromRGBs)

    def forward(self,x,cond,in_level,alpha,label=None):
        rgb=x
        style_feat=self.style_feat_block(cond)
        if self.use_label:
            label_feat=self.label_feat_block(get_label(label,x))
            style_feat=torch.cat([style_feat,label_feat],dim=1)
            style_feat=self.merge_feat_block(style_feat)
        x=self.fromRGBs[in_level](x)
        if in_level>self.start_level:
            level_style=self.level_style_blocks[in_level-1](style_feat)
            x=self.level_blocks[in_level-1](x,level_style)
            if alpha<1 and alpha>0:
                prev_x=self.fromRGBs[in_level-1](F.interpolate(rgb,scale_factor=0.5))
                x=alpha*x+(1-alpha)*prev_x
            for level in reversed(range(self.start_level,in_level-1)):
                level_style=self.level_style_blocks[level](style_feat)
                x=self.level_blocks[level](x,level_style)
        final_style=self.final_style_block(style_feat)
        if self.use_label and self.use_separate_block:
            block_id=0 if label=='hero' else 1
            # print('D',block_id)
            return self.final_block[block_id](x,final_style)
        else:
            return self.final_block(x,final_style)

class projD(nn.Module):
    def __init__(self,start_level=3,end_level=8,input_channels=6,channels=None,cond_size=90,output_size=1,y_size=10,use_coord_conv=True):
        super(self.__class__,self).__init__()
        assert(channels is not None)
        init_size=2**start_level
        self.start_level=start_level
        self.end_level=end_level
        self.channels=channels
        self.input_channels=input_channels

        if use_coord_conv: # only use conv in fromRGBs to save parameters and space
            conv=CoordConv2d
        else:
            conv=Conv2d

        # output size of level_blocks[level] is 2^level
        self.level_blocks=[None for i in range(end_level)]
        for level in range(start_level,end_level):
            self.level_blocks[level]=nn.Sequential(
                Conv2d(3,1,1,channels[level+1],channels[level]),
                nn.InstanceNorm2d(channels[level]),
                LeakyReLU(),
                Conv2d(3,1,1,channels[level],channels[level]),
                nn.InstanceNorm2d(channels[level]),
                LeakyReLU(),
                nn.AvgPool2d(kernel_size=2)
                )
        self.level_blocks=nn.ModuleList(self.level_blocks)
        self.fromRGBs=[None for i in range(end_level+1)]
        for level in range(start_level,end_level+1):
            self.fromRGBs[level]=conv(1,1,0,input_channels,channels[level])
        self.fromRGBs=nn.ModuleList(self.fromRGBs)

        self.feat_block=nn.Sequential(
            BatchStd(),
            Conv2d(3,1,1,channels[start_level]+1,channels[start_level]),
            nn.InstanceNorm2d(channels[start_level]),
            LeakyReLU(),
            Conv2d(init_size,1,0,channels[start_level],channels[start_level]),
            LeakyReLU(),
            Reshape((-1,1,1,channels[start_level])),
            )

        feat_size=channels[start_level]
        feat_channels=[feat_size,16]
        real_block=[]
        for i in range(len(feat_channels)-1):
            real_block+=[Linear(feat_channels[i],feat_channels[i+1]),
                        nn.InstanceNorm2d(feat_channels[i+1]),
                        LeakyReLU()]
        real_block+=[Linear(feat_channels[len(feat_channels)-1],1)]
        self.real_block=nn.Sequential(*real_block)

        cond_channels=[cond_size,40,y_size]
        cond_block=[]
        for i in range(len(cond_channels)-1):
            cond_block+=[Linear(cond_channels[i],cond_channels[i+1]),
                        InstanceNorm1d(cond_channels[i+1]),
                        LeakyReLU()]
        self.cond_block=nn.Sequential(*cond_block)

        self.V=nn.Parameter(torch.Tensor(y_size,feat_size))
        nn.init.normal_(self.V.data)

    def V_prod(self,y,x):
        n_samples=y.size(0)
        y=y.view(n_samples,1,-1)
        x=x.view(n_samples,-1,1)
        V=self.V.view(1,self.V.size(0),self.V.size(1)).repeat(n_samples,1,1)
        return torch.matmul(y,torch.matmul(V,x)).squeeze()/self.V.numel()

    def forward(self,x,cond,in_level,alpha):
        rgb=x
        x=self.fromRGBs[in_level](x)
        if in_level!=self.start_level:
            x=self.level_blocks[in_level-1](x)
            if alpha<1 and alpha>0:
                prev_x=self.fromRGBs[in_level-1](F.interpolate(rgb,scale_factor=0.5))
                x=alpha*x+(1-alpha)*prev_x
            for level in reversed(range(self.start_level,in_level-1)):
                x=self.level_blocks[level](x)
        feat=self.feat_block(x)
        real=self.real_block(feat)
        cond=self.cond_block(cond)
        cond=self.V_prod(cond,feat)
        return real,cond

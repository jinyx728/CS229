######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self,size):
        super(self.__class__,self).__init__()
        self.size=size

    def forward(self,x):
        return x.view(self.size)

class CoordConv(nn.Module):
    def __init__(self,conv_layer,coord_conv_stride,coord_scale=0.1):
        super(self.__class__,self).__init__()
        self.conv_layer=conv_layer
        self.coord_conv_stride=coord_conv_stride
        self.coord_scale=coord_scale

    def forward(self,x):
        N,D,H,W=x.size()
        stride=self.coord_conv_stride
        device=x.device
        coord_scale=self.coord_scale
        x_plane=torch.linspace(-1*coord_scale+stride/2,1*coord_scale-stride/2,W).type(x.dtype).to(device)
        x_plane=x_plane.view(1,1,1,-1).repeat(N,1,H,1)
        y_plane=torch.linspace(-1*coord_scale+stride/2,1*coord_scale-stride/2,H).type(x.dtype).to(device)
        y_plane=y_plane.view(1,1,-1,1).repeat(N,1,1,W)
        x=torch.cat([x,x_plane,y_plane],dim=1)
        return self.conv_layer(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class UpConv(nn.Module):
    def __init__(self,in_channels,out_channels,use_up_conv=False,use_coord_conv=False,coord_conv_stride=0,coord_scale=0.1,n_res_blocks=0):
        super(self.__class__,self).__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels

        main=[]
        for i in range(n_res_blocks):
            main.append(ResBlock(in_channels,in_channels))

        if use_up_conv:
            if use_coord_conv:
                conv=CoordConv(nn.Conv2d(in_channels+2,out_channels,3,1,1),coord_conv_stride/2,coord_scale)
            else:
                conv=nn.Conv2d(in_channels,out_channels,3,1,1)

            main+=[nn.Upsample(scale_factor=2),
                   conv
                ]
        else:
            if use_coord_conv:
                conv_transpose=CoordConv(nn.ConvTranspose2d(in_channels+2,out_channels,4,2,1),coord_conv_stride,coord_scale)
            else:
                conv_transpose=nn.ConvTranspose2d(in_channels,out_channels,4,2,1)
            main+=[conv_transpose]

        self.main=nn.Sequential(*main)


    def forward(self,x):
        return self.main(x)


class ConvDecoder(nn.Module):
    def __init__(self,input_channels=90,init_linear_layers=1,init_linear_channels=128,init_channels=64,init_size=16,output_channels=6,output_size=128,use_up_conv=False,use_coord_conv=False,use_skip_link=False,n_res_blocks=0,use_dropout=False,use_multi_layer_loss=False,conv_out_channels=None,relu_type='relu'):
        super(self.__class__,self).__init__()
        def get_relu(in_channels):
            if relu_type=='prelu':
                return nn.PReLU(in_channels)
            else:
                return nn.ReLU()

        # init linear layers
        if init_linear_layers==0:
            fc_module=[
                View((-1,input_channels,1,1)),
                nn.ConvTranspose2d(input_channels,init_channels,init_size,1,0),
                # nn.Linear(input_channels,init_channels*init_size*init_size),
                # View((-1,init_channels,init_size,init_size)),
            ]
        else:
            fc_module=[
                nn.Linear(input_channels,init_linear_channels),
                nn.BatchNorm1d(init_linear_channels),
                # nn.ReLU()
                get_relu(init_linear_channels)
            ]
            for i in range(init_linear_layers-1):
                fc_module+=[
                    nn.Linear(init_linear_channels,init_linear_channels),
                    nn.BatchNorm1d(init_linear_channels),
                    # nn.ReLU()
                    get_relu(init_linear_channels)
                ]

            fc_module+=[
                nn.Linear(init_linear_channels,init_channels*init_size*init_size),
                View((-1,init_channels,init_size,init_size)),
                nn.BatchNorm2d(init_channels),
            ]

        # conv layers
        conv_module=[]
        size=init_size
        channels=init_channels
        coord_scale=0.1 # [-0.1,0.1]
        first_conv=True
        block_i=0
        while size<output_size:
            in_channels=channels
            if conv_out_channels is not None:
                out_channels=conv_out_channels[block_i]
            else:
                out_channels=channels//2
                if out_channels<output_channels:
                    out_channels=output_channels
            
            if use_skip_link and not first_conv:
                in_channels+=init_channels

            up_block=[]
            if use_dropout and size<=64: # heuristics
                up_block=[nn.Dropout(0.5)]
            up_block+=[
                # nn.ReLU(),
                get_relu(in_channels),
                UpConv(in_channels,out_channels,use_up_conv=use_up_conv,use_coord_conv=use_coord_conv,coord_conv_stride=2*coord_scale/size,coord_scale=coord_scale,n_res_blocks=n_res_blocks),
                nn.BatchNorm2d(out_channels)]

            conv_module.append(
                nn.Sequential(*up_block))

            first_conv=False
            size*=2
            channels=out_channels
            block_i+=1

        # final conv layer
        in_channels=channels
        if use_skip_link:
            in_channels+=init_channels
        if use_coord_conv:
            conv_module.append(nn.Sequential(
                # nn.ReLU(),
                get_relu(in_channels+2),
                CoordConv(nn.Conv2d(in_channels+2,output_channels,3,1,1),coord_conv_stride=2*coord_scale/size,coord_scale=coord_scale)))
        else:
            conv_module.append(nn.Sequential(
                # nn.ReLU(),
                get_relu(in_channels),
                nn.Conv2d(in_channels,output_channels,3,1,1)))

        self.fc_module=nn.Sequential(*fc_module)
        self.conv_module=nn.Sequential(*conv_module)
        self.use_skip_link=use_skip_link

        self.use_multi_layer_loss=use_multi_layer_loss
        if use_multi_layer_loss:
            out_modules=[nn.Sequential(
                        # nn.ReLU(),
                        get_relu(init_channels),
                        nn.Conv2d(init_channels,output_channels,1,1,0)
                    )]
            for i in range(len(conv_module)-2):
                upconv_module=conv_module[i]
                in_channels=upconv_module[1].out_channels
                if use_skip_link:
                    in_channels+=init_channels
                out_channels=output_channels
                out_modules.append(nn.Sequential(
                        # nn.ReLU(),
                        get_relu(in_channels),
                        nn.Conv2d(in_channels,out_channels,1,1,0)
                    ))
            self.out_modules=nn.ModuleList(out_modules)



    def forward(self,x):
        # assert(not (self.use_skip_link and self.use_multi_layer_loss))
        fc_out=self.fc_module(x)
        if self.use_skip_link and self.use_multi_layer_loss:
            out=fc_out
            img_outs=[]
            n_convs=len(self.conv_module)
            for i in range(n_convs):
                if i<n_convs-1: # skip the last output, append the last output
                    img_out=self.out_modules[i](out)
                    img_outs.append(img_out)
                out=self.conv_module[i](out)
                if i!=n_convs-1:
                    out=torch.cat([out,nn.functional.upsample(fc_out,scale_factor=out.size(2)//fc_out.size(2))],dim=1)
            img_outs.append(out)
            return img_outs
        if self.use_skip_link:
            n_convs=len(self.conv_module)
            out=fc_out
            for i in range(n_convs):
                out=self.conv_module[i](out)
                if i!=n_convs-1:
                    out=torch.cat([out,nn.functional.upsample(fc_out,scale_factor=out.size(2)//fc_out.size(2))],dim=1)
            return out
        elif self.use_multi_layer_loss:
            out=fc_out
            img_outs=[]
            n_convs=len(self.conv_module)
            for i in range(n_convs):
                if i<n_convs-1: # skip the last output, append the last output
                    img_out=self.out_modules[i](out)
                    img_outs.append(img_out)
                out=self.conv_module[i](out)
            # append final out
            img_outs.append(out)
            return img_outs
        else:
            return self.conv_module(fc_out)

def xavier_weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m,'weight'):
        torch.nn.init.xavier_normal_(m.weight)
        # m.weight.data.xavier_normal_(0.0, 0.02)

# For backward compatibility
class ConvDecoderV0(nn.Module):
    def __init__(self,input_channels=90,init_linear_layers=1,init_linear_channels=128,init_channels=64,init_size=16,output_channels=6,output_size=128):
        super(self.__class__,self).__init__()
        # init linear layers
        if init_linear_layers==0:
            main=[
                nn.Linear(input_channels,init_channels*init_size*init_size),
                View((-1,init_channels,init_size,init_size)),
            ]
        else:
            main=[
                nn.Linear(input_channels,init_linear_channels),
                nn.BatchNorm1d(init_linear_channels),
                nn.ReLU()
            ]
            for i in range(init_linear_layers-1):
                fc_module+=[
                    nn.Linear(init_linear_channels,init_linear_channels),
                    nn.BatchNorm1d(init_linear_channels),
                    nn.ReLU()
                ]

            main+=[
                nn.Linear(init_linear_channels,init_channels*init_size*init_size),
                View((-1,init_channels,init_size,init_size)),
                nn.BatchNorm2d(init_channels),
            ]

        # conv layers
        size=init_size
        channels=init_channels
        coord_scale=0.1 # [-0.1,0.1]
        while size<output_size:
            in_channels=channels
            out_channels=channels//2

            main+=[nn.ReLU(),
                nn.ConvTranspose2d(in_channels,out_channels,4,2,1),
                nn.BatchNorm2d(out_channels)]

            size*=2
            channels=out_channels

        # final conv layer
        in_channels=channels
        main+=[nn.ReLU(),
            nn.Conv2d(in_channels,output_channels,3,1,1)]

        self.main=nn.Sequential(*main)


    def forward(self,x):
        return self.main(x)

# reference: https://github.com/YadiraF/PRNet/blob/master/predictor.py
class ReLUConvTransposeBN(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride):
        super(self.__class__,self).__init__()
        self.main=nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,padding,stride),
            nn.BatchNorm2d(out_channels)
            )
    def forward(self,x):
        return self.main(x)

class ConvDecoderFace(nn.Module):
    def __init__(self,input_channels=90,init_linear_layers=1,init_linear_channels=128,init_channels=512,init_size=8,output_channels=6,output_size=128):
        super(self.__class__,self).__init__()
        # fc modules
        if init_linear_layers==0:
            fc_module=[
                nn.Linear(input_channels,init_channels*init_size*init_size),
                View((-1,init_channels,init_size,init_size)),
            ]
        else:
            fc_module=[
                nn.Linear(input_channels,init_linear_channels),
                nn.BatchNorm1d(init_linear_channels),
                nn.ReLU()
            ]
            for i in range(init_linear_layers-1):
                fc_module+=[
                    nn.Linear(init_linear_channels,init_linear_channels),
                    nn.BatchNorm1d(init_linear_channels),
                    nn.ReLU()
                ]

            fc_module+=[
                nn.Linear(init_linear_channels,init_channels*init_size*init_size),
                View((-1,init_channels,init_size,init_size)),
                nn.BatchNorm2d(init_channels),
            ]

        # conv modules
        conv_module=[
            ReLUConvTransposeBN(512,512,3,1,1), #8x8x512
            ReLUConvTransposeBN(512,256,4,2,1), #16x16x256
            ReLUConvTransposeBN(256,256,3,1,1), #16x16x256
            ReLUConvTransposeBN(256,256,3,1,1), #16x16x256
            ReLUConvTransposeBN(256,128,4,2,1), #32x32x128
            ReLUConvTransposeBN(128,128,3,1,1), #32x32x128
            ReLUConvTransposeBN(128,128,3,1,1), #32x32x128
            ReLUConvTransposeBN(128,64,4,2,1), #64x64x64
            ReLUConvTransposeBN(64,64,3,1,1), #64x64x64
            ReLUConvTransposeBN(64,64,3,1,1), #64x64x64
            ReLUConvTransposeBN(64,32,4,2,1), #128x128x32
            ReLUConvTransposeBN(32,32,3,1,1), #128x128x32
            ReLUConvTransposeBN(32,16,3,1,1), #128x128x16
            ReLUConvTransposeBN(16,16,3,1,1), #128x128x16
            ReLUConvTransposeBN(16,6,3,1,1), #128x128x6
            ReLUConvTransposeBN(6,6,3,1,1), #128x128x6
            nn.ConvTranspose2d(6,6,3,1,1)
            ]

        self.fc_module=nn.Sequential(*fc_module)
        self.conv_module=nn.Sequential(*conv_module)

    def forward(self,x):
        fc_out=self.fc_module(x)
        conv_out=self.conv_module(fc_out)
        return conv_out

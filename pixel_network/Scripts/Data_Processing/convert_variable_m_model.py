######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../Train_Conv_Mix')
import torch
import torch.nn as nn
from conv_decoder import ConvDecoder
from ctx_utils import ctx

def convert(src_path,tgt_path):
    cp=torch.load(ctx['cp'],map_location={"cuda:{}".format(i):"cuda:0" for i in range(8)})
    src_state_dict_items=cp['state_dict'].items()
    tgt_net=ConvDecoder(ctx['input_size'],
                    init_linear_layers=ctx['init_linear_layers'],
                    output_size=ctx['offset_img_size'] if not ctx['use_patches'] else ctx['crop_size'],
                    use_coord_conv=ctx['use_coord_conv'],
                    use_up_conv=ctx['use_up_conv'],
                    use_skip_link=ctx['use_skip_link'],
                    use_multi_layer_loss=ctx['use_multi_layer_loss'],
                    init_channels=ctx['init_channels'],
                    init_size=ctx['init_size'],
                    n_res_blocks=ctx['n_res_blocks'],
                    use_dropout=ctx['use_dropout'],
                    output_channels=6+1 if not ctx['use_patches'] else 3+1,
                    relu_type=ctx['relu'])
    tgt_state_dict=tgt_net.state_dict()
    n=len(src_state_dict_items)
    i=0
    for k,v in src_state_dict_items:
        if i==n-2:
            print('alter:',k,v.size())
            tgt_state_dict[k][:-1,:,:,:].copy_(v.data)
        elif i==n-1:
            print('alter:',k,v.size())
            tgt_state_dict[k][:-1].copy_(v.data)
        else:
            print(k,v.size())
            tgt_state_dict[k].copy_(v.data)
        i+=1
    cp['state_dict']=tgt_state_dict
    torch.save(tgt_state_dict,tgt_path)



if __name__=='__main__':
    convert('../../rundir/lowres/xyz/saved_models/train_model_best.pth.tar','../../rundir/lowres/variable_m/saved_models/train_model_best.pth.tar')


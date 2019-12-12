######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import shutil
from os.path import join,isfile,isdir

def collect_pd_objs(eval_dir,out_dir,pattern):
    if not isdir(out_dir):
        os.makedirs(out_dir)
    dirs=os.listdir(eval_dir)
    for d in dirs:
        sample_id=int(d)
        src_path=join(eval_dir,d,pattern.format(sample_id))
        tgt_path=join(out_dir,'{:08d}.obj'.format(sample_id))
        print('copy to',tgt_path)
        shutil.copyfile(src_path,tgt_path)

if __name__=='__main__':
    # collect_pd_objs('/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/Learning/rundir/lowres_normal/pd_seq2/eval_pred','/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/joint_data/seq2/videos/collected_objs')
    collect_pd_objs('../../rundir/lowres_cudaqs/uvn/eval_test','stats_test/cudaqs/cr','cr_cloth_{:08d}.obj')

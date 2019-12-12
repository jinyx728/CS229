######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
import shutil
import argparse

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('-samples_dir',default=None,type=str)
    parser.add_argument('-out_dir',default=None,type=str)
    parser.add_argument('-convert_to_tri',action='store_true')
    parser.add_argument('-prefix',default=None)
    parser.add_argument('-test_case',choices=['whole','regions','whole_gt','prefix'],default='prefix')

    args=parser.parse_args()

    # samples_dir='/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/Learning/rundir/lowres_normal_regions/eval_test_test'
    samples_dir=args.samples_dir
    assert(samples_dir is not None)
    # out_dir='out_objs/lowres_normal_regions'
    out_dir=args.out_dir
    if out_dir is None:
        out_dir=os.path.join(samples_dir,'tris')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    convert_to_tri=args.convert_to_tri

    sample_dirs=os.listdir(samples_dir)

    for sample_dir in sample_dirs:
        try:
            test_id=int(sample_dir)
        except ValueError as err:
            continue
        print(sample_dir)
        if args.test_case=='regions':
            src_path=os.path.join(samples_dir,sample_dir,'stitched_regions.obj')
        elif args.test_case=='whole_gt':
            src_path=os.path.join(samples_dir,sample_dir,'gt_cloth_{:08d}.obj'.format(test_id))
        elif args.test_case=='prefix':
            assert(args.prefix is not None)
            src_path=os.path.join(samples_dir,sample_dir,'{}_{:08d}.obj'.format(args.prefix,test_id))
        else:
            src_path=os.path.join(samples_dir,sample_dir,'cloth_{:08d}.obj'.format(test_id))
        if convert_to_tri:
            tri_path=os.path.join(out_dir,'{:08d}.tri'.format(test_id))
            cmd='$PHYSBAM/Tools/obj2tri/obj2tri {} {}'.format(src_path,tri_path)
            os.system(cmd)
        else:   
            tgt_path=os.path.join(out_dir,'{:08d}.obj'.format(test_id))
            # print('src',src_path,'tgt',tgt_path)
            shutil.copy(src_path,tgt_path)

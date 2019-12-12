######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
samples_dir='/data/zhenglin/PhysBAM/Private_Projects/cloth_on_virtual_body/Learning/rundir/lowres_normal_regions/eval_test_test'
dirs=os.listdir(samples_dir)
for d in dirs:
    os.chdir(os.path.join(samples_dir,d))
    os.system('meshlab stitched_regions.obj &')
    # break
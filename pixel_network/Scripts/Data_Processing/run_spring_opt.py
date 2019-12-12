######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os

def run_spring_opt(start,end,step):
    r=[]
    c=start
    while True:
        if c+step-1>=end:
            r.append((c,end))
            break
        r.append((c,c+step-1))
        c+=step
    for start,end in r:
        cmd='python test_spring_opt.py -start {} -end {} &'.format(start,end)
        print(cmd)
        os.system(cmd)

if __name__=='__main__':
    run_spring_opt(0,2248,60)

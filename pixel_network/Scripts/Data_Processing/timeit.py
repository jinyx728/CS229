######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import time
from collections import defaultdict

def timeit(func):
    def wrapper(*args,**kwargs):
        start=time.time()
        ret=func(*args,**kwargs)
        end=time.time()
        print(func.__name__,end-start,'s')
        return ret
    return wrapper

stat=defaultdict(int)
use_time_stat=True
def timestat(func):
    def wrapper(*args,**kwargs):
        start=time.time()
        ret=func(*args,**kwargs)
        end=time.time()
        if use_time_stat:
            stat[func.__name__]+=end-start
        return ret
    return wrapper

def print_stat():
    if not use_time_stat:
        return
    items=stat.items()
    items=sorted(items,key=lambda v:v[1],reverse=True)
    for k,v in items:
        print(k,v,'s')

def clear_stat():
    stat=defaultdict(int)
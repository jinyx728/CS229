######################################################################
# Copyright 2019. Dan Johnson.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import cudaqs
import time
import sys

def test_row_sum(size):
    cuda0 = torch.device("cuda:0")
    x = torch.rand((size, size), device=cuda0)
    y = cudaqs.row_sum(x)

def test_row_sum_ctpl(size, num_ctpl_threads):
    cuda0 = torch.device("cuda:0")
    x = torch.rand((size, size), device=cuda0)
    y = cudaqs.row_sum_ctpl(x, num_ctpl_threads)

def row_sum(size):
    cuda0 = torch.device("cuda:0")
    x = torch.rand((size, size), device=cuda0)
    y = torch.sum(x, dim=0)

if __name__=='__main__':
    test_row_sum(1)
    test_row_sum_ctpl(4, 2)
    row_sum(1)
    sizes = [5, 50, 500, 5000, 50000]
    for size in sizes:
        t0 = time.time()
        test_row_sum(size)
        t1 = time.time()
        num_ctpl_threads = 32
        test_row_sum_ctpl(size, num_ctpl_threads)
        t2 = time.time()
        row_sum(size)
        t3 = time.time()
        print("size: "+str(size))
        print("cuda: "+str(t1-t0))
        print("cuda with ctpl: "+str(t2-t1))
        print("torch: "+str(t3-t2))
        print("")

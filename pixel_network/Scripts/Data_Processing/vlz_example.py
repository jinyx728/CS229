######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
from physbam_example import Example
import numpy as np
class VlzExample(Example):
    def __init__(self,output_directory):
        super(VlzExample,self).__init__(output_directory)

    def draw_edges(self,v,edges,restlengths,netural_color=np.array([1,1,1]),compress_color=np.array([0,0,1]),stretch_color=np.array([1,0,0]),quantile=10,bin_size=0.05):
        line_width=8
        lengths=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
        ratios=lengths/restlengths
        diff_ratios=ratios-1
        abs_diff=np.abs(diff_ratios)
        ts=np.floor(abs_diff/bin_size)/quantile
        ts[ts>1]=1
        n_edges=len(ratios)
        for i in range(n_edges):
            t=ts[i]
            diff_ratio=diff_ratios[i]
            other_color=compress_color if diff_ratio<0 else stretch_color
            color=(1-t)*netural_color+t*other_color
            self.proto_debug_utils.Add_Line(v[edges[i,0]],v[edges[i,1]],color=color,width=line_width)

    def draw_lines(self,vs,lines,line_colors):
        line_width=8
        for line_i in range(len(lines)):
            i0,i1=lines[line_i]
            color=line_colors[line_i]
            v0,v1=vs[i0],vs[i1]
            self.proto_debug_utils.Add_Line(v0,v1,color=color,width=line_width)
        for v in vs:
            self.proto_debug_utils.Add_Point(v,size=8,color=[0.0,1.0,0.0])

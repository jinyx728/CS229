######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
from collections import defaultdict

def get_linear_edges(fcs):
	n_vts=np.max(fcs)+1
	nbs=[set() for i in range(n_vts)]
	for fc in fcs:
	    nbs[fc[0]].add(fc[1])
	    nbs[fc[0]].add(fc[2])
	    nbs[fc[1]].add(fc[0])
	    nbs[fc[1]].add(fc[2])
	    nbs[fc[2]].add(fc[0])
	    nbs[fc[2]].add(fc[1])

	edges=[]
	for i in range(n_vts):
	    nb=nbs[i]
	    for n in nb:
	        if n<i:
	            continue
	        edges.append([i,n])
	return np.array(edges).astype(np.int32)

def get_bend_edges(fcs):
	n_vts=np.max(fcs)+1
	def hash_edge(v1,v2):
		if v1<=v2:
			return v1*n_vts+v2
		else:
			return v2*n_vts+v1
	axial_pairs=defaultdict(list)
	for fc in fcs:
		i0,i1,i2=fc[0],fc[1],fc[2]
		axial_pairs[hash_edge(i0,i1)].append(i2)
		axial_pairs[hash_edge(i1,i2)].append(i0)
		axial_pairs[hash_edge(i2,i0)].append(i1)
	edges=[]
	for k,v in axial_pairs.items():
		if len(v)==2:
			edges.append(v)
		elif len(v)!=1:
			assert(False)
	return np.array(edges).astype(np.int32)

def filter_edges(edges,front_vt_ids,back_vt_ids):
	front_vt_set=set(front_vt_ids.tolist())
	back_vt_set=set(back_vt_ids.tolist())
	filtered_edges=[]
	for edge in edges:
		if edge[0] in front_vt_set and edge[1] in front_vt_set:
			filtered_edges.append(edge)
		if edge[0] in back_vt_set and edge[1] in back_vt_set:
			filtered_edges.append(edge)
	return np.array(filtered_edges)


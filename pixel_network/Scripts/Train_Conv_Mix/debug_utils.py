######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import torch
import gc
def print_tensors():
	print('print tensors')
	total_numel=0
	for obj in gc.get_objects():
		try:
		    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
		    	print(obj.device,obj.type(), obj.size(),'nele',obj.numel())
		    	total_numel+=obj.numel()
		except Exception as e:
			print('Exception')
	print('total_numel',total_numel)
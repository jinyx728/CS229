######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import numpy as np
import os
from os.path import join,isdir

class FileLogger:
	def __init__(self,log_dir):
		self.log_dir=log_dir
		if not isdir(log_dir):
			os.makedirs(log_dir)
		os.system('rm {}/*.txt'.format(log_dir))
		self.file_handles={}

	def log(self,name,value,step):
		if name in self.file_handles:
			file=self.file_handles[name]
		else:
			file=open(join(self.log_dir,'{}.txt'.format(name)),'w')
			self.file_handles[name]=file
		file.write('{} {}\n'.format(step,value))
		file.flush()
		os.fsync(file)

	def close(self):
		for _,f in self.file_handles.items():
			f.close()
		self.file_handles={}
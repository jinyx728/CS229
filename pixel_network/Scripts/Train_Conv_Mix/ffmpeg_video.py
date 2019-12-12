######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import os
from os.path import join,isfile,isdir

def ffmpeg_video(in_dir,out_dir):
	if not isdir(out_dir):
		os.makedirs(out_dir)
	tester_dirs=os.listdir(in_dir)
	tester_dirs.sort()
	for tester_dir in tester_dirs:
		png_dir=join(in_dir,tester_dir,'pd_00')
		out_path=join(out_dir,'{}.mp4'.format(tester_dir))
		cmd='ffmpeg -i {}/frame_%04d_blend.png -c:v libxvid -q:v 1 {}'.format(png_dir,out_path)
		os.system(cmd)
		# break

if __name__=='__main__':
	ffmpeg_video('Test1_seq','Test1_mp4')
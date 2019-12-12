######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
import sys
sys.path.insert(0,'../../../../opengl_3d_proto/Proto/Python')
from PROTO_DEBUG_UTILS import PROTO_DEBUG_UTILS
import os
from os.path import join,isdir,isfile

class Example:
    def __init__(self,output_directory,first_frame=0):
        self.proto_debug_utils=PROTO_DEBUG_UTILS()
        self.output_directory=output_directory
        self.current_frame=first_frame
        self.first_frame=first_frame
        if not isdir(self.output_directory):
            os.makedirs(self.output_directory)
        common_dir=join(self.output_directory,'common')
        if not isdir(common_dir):
            os.makedirs(common_dir)
        self.write_first_frame(first_frame)
        self.write_output_files(first_frame)

    def add_frame(self):
        self.current_frame+=1
        self.write_last_frame(self.current_frame)
        self.write_output_files(self.current_frame)

    def write_first_frame(self,frame):
        path=join(self.output_directory,'common/first_frame')
        with open(path,'w') as f:
            f.write('{}\n'.format(frame))

    def write_last_frame(self,frame):
        path=join(self.output_directory,'common/last_frame')
        with open(path,'w') as f:
            f.write('{}\n'.format(frame))

    def write_output_files(self,frame):
        frame_dir=join(self.output_directory,'{}'.format(frame))
        if not isdir(frame_dir):
            os.makedirs(frame_dir)
        self.proto_debug_utils.Write_Output_Files(self.output_directory,frame)
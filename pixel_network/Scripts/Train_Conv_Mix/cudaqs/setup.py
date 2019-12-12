######################################################################
# Copyright 2019. Dan Johnson, Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='cudaqs',
      ext_modules=[CUDAExtension('cudaqs', ['src/cudaqs_python.cpp'], libraries=["SRC_LIB"], library_dirs=["lib_build/src"],extra_compile_flags=['-fPIC'])],
      cmdclass={'build_ext': BuildExtension})

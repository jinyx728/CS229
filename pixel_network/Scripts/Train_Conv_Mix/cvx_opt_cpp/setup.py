######################################################################
# Copyright 2019. Zhenglin Geng.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, include_paths
import os
from os.path import join
ecos_dir='../ecos'
eigen_dir='../eigen'
cmk_dir='build_cmk'
# cmk_dir='build_dbg'
print(include_paths())
print('cmk_dir',cmk_dir)

# setup(name='cvx_opt_cpp',
#       ext_modules=[CppExtension('cvx_opt_cpp', ['cvx_opt_py.cpp','cvx_opt_utils.cpp','cvx_opt_forward.cpp','cvx_opt_backward.cpp','ecos_opt.cpp','forward_opt.cpp','backward_opt.cpp'],
#         include_dirs=[join(ecos_dir,'include'),join(ecos_dir,'external/ldl/include'),join(ecos_dir,'external/amd/include'),join(ecos_dir,'external/SuiteSparse_config'),eigen_dir],
#         libraries=['ecos'],
#         library_dirs=[ecos_dir],
#       	extra_compile_args=['-DLDL_LONG', '-DDLONG'])],
#       cmdclass={'build_ext': BuildExtension})

setup(name='cvx_opt_cpp',
      ext_modules=[CppExtension('cvx_opt_cpp', ['cvx_opt_py.cpp'],
        include_dirs=[join(ecos_dir,'include'),join(ecos_dir,'external/ldl/include'),join(ecos_dir,'external/amd/include'),join(ecos_dir,'external/SuiteSparse_config'),eigen_dir],
        libraries=['cvx_opt','ecos'],
        library_dirs=[cmk_dir,join(ecos_dir,'build_cmk')],
        extra_compile_args=['-DLDL_LONG', '-DDLONG','-fPIC'])],
      cmdclass={'build_ext': BuildExtension})
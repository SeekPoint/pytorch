from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension

working_dirs = r'C:\yk_repo\pytorch\pt_ext_01'

# 头文件目录
include_dirs = os.path.dirname(os.path.abspath(__file__))
#源代码目录
source_file = glob.glob(os.path.join(working_dirs, 'src', '*.cpp'))

setup(
    name='test_cpp',  # 模块名称
    ext_modules=[CppExtension('test_cpp', sources=source_file, include_dirs=[include_dirs])],
    cmdclass={
        'build_ext': BuildExtension
    }
)
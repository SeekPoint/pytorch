# Welcome to the PyTorch setup.py.
#
# Environment variables you are probably interested in:
#
#   DEBUG
#     build with -O0 and -g (debug symbols)
#
#   REL_WITH_DEB_INFO
#     build with optimizations and -g (debug symbols)
#
#   MAX_JOBS
#     maximum number of compile jobs we should use to compile your code
#
#   USE_CUDA=0
#     disables CUDA build
#
#   CFLAGS
#     flags to apply to both C and C++ files to be compiled (a quirk of setup.py
#     which we have faithfully adhered to in our build system is that CFLAGS
#     also applies to C++ files (unless CXXFLAGS is set), in contrast to the
#     default behavior of autogoo and cmake build systems.)
#
#   CC
#     the C/C++ compiler to use
#
# Environment variables for feature toggles:
#
#   USE_CUDNN=0
#     disables the cuDNN build
#
#   USE_FBGEMM=0
#     disables the FBGEMM build
#
#   USE_KINETO=0
#     disables usage of libkineto library for profiling
#
#   USE_NUMPY=0
#     disables the NumPy build
#
#   BUILD_TEST=0
#     disables the test build
#
#   USE_MKLDNN=0
#     disables use of MKLDNN
#
#   USE_MKLDNN_ACL
#     enables use of Compute Library backend for MKLDNN on Arm;
#     USE_MKLDNN must be explicitly enabled.
#
#   MKLDNN_CPU_RUNTIME
#     MKL-DNN threading mode: TBB or OMP (default)
#
#   USE_STATIC_MKL
#     Prefer to link with MKL statically - Unix only
#   USE_ITT=0
#     disable use of Intel(R) VTune Profiler's ITT functionality
#
#   USE_NNPACK=0
#     disables NNPACK build
#
#   USE_QNNPACK=0
#     disables QNNPACK build (quantized 8-bit operators)
#
#   USE_DISTRIBUTED=0
#     disables distributed (c10d, gloo, mpi, etc.) build
#
#   USE_TENSORPIPE=0
#     disables distributed Tensorpipe backend build
#
#   USE_GLOO=0
#     disables distributed gloo backend build
#
#   USE_MPI=0
#     disables distributed MPI backend build
#
#   USE_SYSTEM_NCCL=0
#     disables use of system-wide nccl (we will use our submoduled
#     copy in third_party/nccl)
#
#   BUILD_CAFFE2_OPS=0
#     disable Caffe2 operators build
#
#   BUILD_CAFFE2=0
#     disable Caffe2 build
#
#   USE_IBVERBS
#     toggle features related to distributed support
#
#   USE_OPENCV
#     enables use of OpenCV for additional operators
#
#   USE_OPENMP=0
#     disables use of OpenMP for parallelization
#
#   USE_FFMPEG
#     enables use of ffmpeg for additional operators
#
#   USE_FLASH_ATTENTION=0
#     disables building flash attention for scaled dot product attention
#
#   USE_LEVELDB
#     enables use of LevelDB for storage
#
#   USE_LMDB
#     enables use of LMDB for storage
#
#   BUILD_BINARY
#     enables the additional binaries/ build
#
#   ATEN_AVX512_256=TRUE
#     ATen AVX2 kernels can use 32 ymm registers, instead of the default 16.
#     This option can be used if AVX512 doesn't perform well on a machine.
#     The FBGEMM library also uses AVX512_256 kernels on Xeon D processors,
#     but it also has some (optimized) assembly code.
#
#   PYTORCH_BUILD_VERSION
#   PYTORCH_BUILD_NUMBER
#     specify the version of PyTorch, rather than the hard-coded version
#     in this file; used when we're building binaries for distribution
#
#   TORCH_CUDA_ARCH_LIST
#     specify which CUDA architectures to build for.
#     ie `TORCH_CUDA_ARCH_LIST="6.0;7.0"`
#     These are not CUDA versions, instead, they specify what
#     classes of NVIDIA hardware we should generate PTX for.
#
#   PYTORCH_ROCM_ARCH
#     specify which AMD GPU targets to build for.
#     ie `PYTORCH_ROCM_ARCH="gfx900;gfx906"`
#
#   ONNX_NAMESPACE
#     specify a namespace for ONNX built here rather than the hard-coded
#     one in this file; needed to build with other frameworks that share ONNX.
#
#   BLAS
#     BLAS to be used by Caffe2. Can be MKL, Eigen, ATLAS, FlexiBLAS, or OpenBLAS. If set
#     then the build will fail if the requested BLAS is not found, otherwise
#     the BLAS will be chosen based on what is found on your system.
#
#   MKL_THREADING
#     MKL threading mode: SEQ, TBB or OMP (default)
#
#   USE_REDIS
#     Whether to use Redis for distributed workflows (Linux only)
#
#   USE_ZSTD
#     Enables use of ZSTD, if the libraries are found
#
# Environment variables we respect (these environment variables are
# conventional and are often understood/set by other software.)
#
#   CUDA_HOME (Linux/OS X)
#   CUDA_PATH (Windows)
#     specify where CUDA is installed; usually /usr/local/cuda or
#     /usr/local/cuda-x.y
#   CUDAHOSTCXX
#     specify a different compiler than the system one to use as the CUDA
#     host compiler for nvcc.
#
#   CUDA_NVCC_EXECUTABLE
#     Specify a NVCC to use. This is used in our CI to point to a cached nvcc
#
#   CUDNN_LIB_DIR
#   CUDNN_INCLUDE_DIR
#   CUDNN_LIBRARY
#     specify where cuDNN is installed
#
#   MIOPEN_LIB_DIR
#   MIOPEN_INCLUDE_DIR
#   MIOPEN_LIBRARY
#     specify where MIOpen is installed
#
#   NCCL_ROOT
#   NCCL_LIB_DIR
#   NCCL_INCLUDE_DIR
#     specify where nccl is installed
#
#   NVTOOLSEXT_PATH (Windows only)
#     specify where nvtoolsext is installed
#
#   ACL_ROOT_DIR
#     specify where Compute Library is installed
#
#   LIBRARY_PATH
#   LD_LIBRARY_PATH
#     we will search for libraries in these paths
#
#   ATEN_THREADING
#     ATen parallel backend to use for intra- and inter-op parallelism
#     possible values:
#       OMP - use OpenMP for intra-op and native backend for inter-op tasks
#       NATIVE - use native thread pool for both intra- and inter-op tasks
#       TBB - using TBB for intra- and native thread pool for inter-op parallelism
#
#   USE_TBB
#      enable TBB support
#
#   USE_SYSTEM_TBB
#      Use system-provided Intel TBB.
#
#   USE_SYSTEM_LIBS (work in progress)
#      Use system-provided libraries to satisfy the build dependencies.
#      When turned on, the following cmake variables will be toggled as well:
#        USE_SYSTEM_CPUINFO=ON USE_SYSTEM_SLEEF=ON BUILD_CUSTOM_PROTOBUF=OFF

# This future is needed to print Python2 EOL message

'''
本文以PyTorch 1.0为基础。PyTorch的编译首先是python风格的编译，使用了python的setuptools编译系统。以最基本的编译安装命令python setup.py install 为例，这一编译过程包含了如下几个主要阶段：
1，setup.py入口；

2，提前检查依赖项；

3，使用cmake生成Makefile；

4，Make命令——中间源文件的产生；

5，Make命令——编译三方库；

6，Make命令——生成静态库、动态库、可执行文件；

7，Make命令——拷贝文件到合适路径下；

8，setuptools之build_py；

9，setuptools之build_ext；

10，setuptools之install_lib。
'''
from __future__ import print_function
import sys
from pydebug import debuginfo
if sys.version_info < (3,):
    print("Python 2 has reached end-of-life and is no longer supported by PyTorch.")
    sys.exit(-1)
if sys.platform == 'win32' and sys.maxsize.bit_length() == 31:
    print("32-bit Windows Python runtime is not supported. Please switch to 64-bit Python.")
    sys.exit(-1)

import platform
python_min_version = (3, 8, 0)
python_min_version_str = '.'.join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print("You are using Python {}. Python >={} is required.".format(platform.python_version(),
                                                                     python_min_version_str))
    sys.exit(-1)

from setuptools import setup, Extension, find_packages
from collections import defaultdict
from setuptools.dist import Distribution
import setuptools.command.build_ext
import setuptools.command.install
import setuptools.command.sdist
import filecmp
import shutil
import subprocess
import os
import json
import glob
import importlib
import time
import sysconfig

from tools.build_pytorch_libs import build_caffe2
from tools.setup_helpers.env import (IS_WINDOWS, IS_DARWIN, IS_LINUX,
                                     build_type)
from tools.setup_helpers.cmake import CMake
from tools.generate_torch_version import get_torch_version

################################################################################
# Parameters parsed from environment
################################################################################

VERBOSE_SCRIPT = True
RUN_BUILD_DEPS = True
# see if the user passed a quiet flag to setup.py arguments and respect
# that in our parts of the build
EMIT_BUILD_WARNING = False
RERUN_CMAKE = False
CMAKE_ONLY = False
filtered_args = []
for i, arg in enumerate(sys.argv):
    if arg == '--cmake':
        RERUN_CMAKE = True
        continue
    if arg == '--cmake-only':
        # Stop once cmake terminates. Leave users a chance to adjust build
        # options.
        CMAKE_ONLY = True
        continue
    if arg == 'rebuild' or arg == 'build':
        arg = 'build'  # rebuild is gone, make it build
        EMIT_BUILD_WARNING = True
    if arg == "--":
        filtered_args += sys.argv[i:]
        break
    if arg == '-q' or arg == '--quiet':
        VERBOSE_SCRIPT = False
    if arg in ['clean', 'egg_info', 'sdist']:
        RUN_BUILD_DEPS = False
    filtered_args.append(arg)
sys.argv = filtered_args

if VERBOSE_SCRIPT:
    def report(*args):
        print(*args)
else:
    def report(*args):
        pass

    # Make distutils respect --quiet too
    setuptools.distutils.log.warn = report

# Constant known variables used throughout this file
cwd = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(cwd, "torch", "lib")
third_party_path = os.path.join(cwd, "third_party")
caffe2_build_dir = os.path.join(cwd, "build")

# CMAKE: full path to python library
if IS_WINDOWS:
    cmake_python_library = "{}/libs/python{}.lib".format(
        sysconfig.get_config_var("prefix"),
        sysconfig.get_config_var("VERSION"))
    # Fix virtualenv builds
    if not os.path.exists(cmake_python_library):
        cmake_python_library = "{}/libs/python{}.lib".format(
            sys.base_prefix,
            sysconfig.get_config_var("VERSION"))
else:
    cmake_python_library = "{}/{}".format(
        sysconfig.get_config_var("LIBDIR"),
        sysconfig.get_config_var("INSTSONAME"))

cmake_python_include_dir = sysconfig.get_path("include")

'''
setup.py入口
pytorch的编译是从下面这条命令开始的：

python setup.py install 
这是典型的Python convention，使用python setuptools包中的setup函数来进行install。setup函数接收了几个关键的东西：
'''
################################################################################
# Version, create_version_file, and package_name
################################################################################
package_name = os.getenv('TORCH_PACKAGE_NAME', 'torch') #package的名字，从环境变量里取TORCH_PACKAGE_NAME的值，取不到的话就是"torch";
package_type = os.getenv('PACKAGE_TYPE', 'wheel')
version = get_torch_version()
report("Building wheel {}-{}".format(package_name, version))

cmake = CMake()


def get_submodule_folders():
    git_modules_path = os.path.join(cwd, ".gitmodules")
    default_modules_path = [os.path.join(third_party_path, name) for name in [
                            "gloo", "cpuinfo", "tbb", "onnx",
                            "foxi", "QNNPACK", "fbgemm", "cutlass"
                            ]]
    if not os.path.exists(git_modules_path):
        return default_modules_path
    with open(git_modules_path) as f:
        return [os.path.join(cwd, line.split("=", 1)[1].strip()) for line in
                f.readlines() if line.strip().startswith("path")]


def check_submodules():
    def check_for_files(folder, files):
        if not any(os.path.exists(os.path.join(folder, f)) for f in files):
            report("Could not find any of {} in {}".format(", ".join(files), folder))
            report("Did you run 'git submodule update --init --recursive'?")
            sys.exit(1)

    def not_exists_or_empty(folder):
        return not os.path.exists(folder) or (os.path.isdir(folder) and len(os.listdir(folder)) == 0)

    if bool(os.getenv("USE_SYSTEM_LIBS", False)):
        return
    folders = get_submodule_folders()
    debuginfo("yk==folders:", folders)
    # If none of the submodule folders exists, try to initialize them
    if all(not_exists_or_empty(folder) for folder in folders):
        try:
            print(' --- Trying to initialize submodules')
            start = time.time()
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=cwd)
            end = time.time()
            print(' --- Submodule initialization took {:.2f} sec'.format(end - start))
        except Exception:
            print(' --- Submodule initalization failed')
            print('Please run:\n\tgit submodule update --init --recursive')
            sys.exit(1)
    for folder in folders:
        #1，检查以下三方库的CMakeLists.txt:
        check_for_files(folder, ["CMakeLists.txt", "Makefile", "setup.py", "LICENSE", "LICENSE.md", "LICENSE.txt"])
    check_for_files(os.path.join(third_party_path, 'fbgemm', 'third_party',
                                 'asmjit'), ['CMakeLists.txt'])
    check_for_files(os.path.join(third_party_path, 'onnx', 'third_party',
                                 'benchmark'), ['CMakeLists.txt'])


# Windows has very bad support for symbolic links.
# Instead of using symlinks, we're going to copy files over
def mirror_files_into_torchgen():
    # (new_path, orig_path)
    # Directories are OK and are recursively mirrored.
    paths = [
        ('torchgen/packaged/ATen/native/native_functions.yaml', 'aten/src/ATen/native/native_functions.yaml'),
        ('torchgen/packaged/ATen/native/tags.yaml', 'aten/src/ATen/native/tags.yaml'),
        ('torchgen/packaged/ATen/templates', 'aten/src/ATen/templates'),
    ]
    for new_path, orig_path in paths:
        # Create the dirs involved in new_path if they don't exist
        if not os.path.exists(new_path):
            os.makedirs(os.path.dirname(new_path), exist_ok=True)

        # Copy the files from the orig location to the new location
        if os.path.isfile(orig_path):
            debuginfo(f'yk==copy {orig_path} to {new_path}')
            shutil.copyfile(orig_path, new_path)
            continue
        if os.path.isdir(orig_path):
            if os.path.exists(new_path):
                # copytree fails if the tree exists already, so remove it.
                shutil.rmtree(new_path)
            debuginfo(f'yk==copy {orig_path} to {new_path}')
            shutil.copytree(orig_path, new_path)
            continue
        raise RuntimeError("Check the file paths in `mirror_files_into_torchgen()`")

# all the work we need to do _before_ setup runs

def build_deps():
    '''
    build_dep()函数也是检查工作的一部分，其内部包含检查分模块，检查python包依赖以及python文件位置等等。
    值得注意的是，这些检查行为都发生在真正的setup操作之前，目的在于保证安装的正确进行，是安装pytorch必不可少的操作。
    '''
    report('-- Building version ' + version)

    check_submodules()
    ##检查yaml、pyyaml是否能够import。
    check_pydep('yaml', 'pyyaml')

    build_caffe2(version=version,
                 cmake_python_library=cmake_python_library,
                 build_python=True,
                 rerun_cmake=RERUN_CMAKE,
                 cmake_only=CMAKE_ONLY,
                 cmake=cmake)

    if CMAKE_ONLY:
        report('Finished running cmake. Run "ccmake build" or '
               '"cmake-gui build" to adjust build options and '
               '"python setup.py install" to build.')
        sys.exit()

    # Use copies instead of symbolic files.
    # Windows has very poor support for them.
    sym_files = [
        'tools/shared/_utils_internal.py',
        'torch/utils/benchmark/utils/valgrind_wrapper/callgrind.h',
        'torch/utils/benchmark/utils/valgrind_wrapper/valgrind.h',
    ]
    orig_files = [
        'torch/_utils_internal.py',
        'third_party/valgrind-headers/callgrind.h',
        'third_party/valgrind-headers/valgrind.h',
    ]
    for sym_file, orig_file in zip(sym_files, orig_files):
        same = False
        if os.path.exists(sym_file):
            if filecmp.cmp(sym_file, orig_file):
                same = True
            else:
                os.remove(sym_file)
        if not same:
            debuginfo(f'yk==copy {orig_file} to {sym_file}')
            shutil.copyfile(orig_file, sym_file)

################################################################################
# Building dependent libraries
################################################################################

missing_pydep = '''
Missing build dependency: Unable to `import {importname}`.
Please install it via `conda install {module}` or `pip install {module}`
'''.strip()


def check_pydep(importname, module):
    try:
        importlib.import_module(importname)
    except ImportError as e:
        raise RuntimeError(missing_pydep.format(importname=importname, module=module)) from e


class build_ext(setuptools.command.build_ext.build_ext):

    # Copy libiomp5.dylib inside the wheel package on OS X
    def _embed_libiomp(self):

        lib_dir = os.path.join(self.build_lib, 'torch', 'lib')
        libtorch_cpu_path = os.path.join(lib_dir, 'libtorch_cpu.dylib')
        if not os.path.exists(libtorch_cpu_path):
            return
        # Parse libtorch_cpu load commands
        otool_cmds = subprocess.check_output(['otool', '-l', libtorch_cpu_path]).decode('utf-8').split('\n')
        rpaths, libs = [], []
        for idx, line in enumerate(otool_cmds):
            if line.strip() == 'cmd LC_LOAD_DYLIB':
                lib_name = otool_cmds[idx + 2].strip()
                assert lib_name.startswith('name ')
                libs.append(lib_name.split(' ', 1)[1].rsplit('(', 1)[0][:-1])

            if line.strip() == 'cmd LC_RPATH':
                rpath = otool_cmds[idx + 2].strip()
                assert rpath.startswith('path ')
                rpaths.append(rpath.split(' ', 1)[1].rsplit('(', 1)[0][:-1])

        omp_lib_name = 'libiomp5.dylib'
        if os.path.join('@rpath', omp_lib_name) not in libs:
            return

        # Copy libiomp5 from rpath locations
        for rpath in rpaths:
            source_lib = os.path.join(rpath, omp_lib_name)
            if not os.path.exists(source_lib):
                continue
            target_lib = os.path.join(self.build_lib, 'torch', 'lib', omp_lib_name)
            self.copy_file(source_lib, target_lib)
            break

    def run(self):
        # Report build options. This is run after the build completes so # `CMakeCache.txt` exists and we can get an
        # accurate report on what is used and what is not.
        cmake_cache_vars = defaultdict(lambda: False, cmake.get_cmake_cache_variables())
        if cmake_cache_vars['USE_NUMPY']:
            report('-- Building with NumPy bindings')
        else:
            report('-- NumPy not found')
        if cmake_cache_vars['USE_CUDNN']:
            report('-- Detected cuDNN at ' +
                   cmake_cache_vars['CUDNN_LIBRARY'] + ', ' + cmake_cache_vars['CUDNN_INCLUDE_DIR'])
        else:
            report('-- Not using cuDNN')
        if cmake_cache_vars['USE_CUDA']:
            report('-- Detected CUDA at ' + cmake_cache_vars['CUDA_TOOLKIT_ROOT_DIR'])
        else:
            report('-- Not using CUDA')
        if cmake_cache_vars['USE_MKLDNN']:
            report('-- Using MKLDNN')
            if cmake_cache_vars['USE_MKLDNN_ACL']:
                report('-- Using Compute Library for the Arm architecture with MKLDNN')
            else:
                report('-- Not using Compute Library for the Arm architecture with MKLDNN')
            if cmake_cache_vars['USE_MKLDNN_CBLAS']:
                report('-- Using CBLAS in MKLDNN')
            else:
                report('-- Not using CBLAS in MKLDNN')
        else:
            report('-- Not using MKLDNN')
        if cmake_cache_vars['USE_NCCL'] and cmake_cache_vars['USE_SYSTEM_NCCL']:
            report('-- Using system provided NCCL library at {}, {}'.format(cmake_cache_vars['NCCL_LIBRARIES'],
                                                                            cmake_cache_vars['NCCL_INCLUDE_DIRS']))
        elif cmake_cache_vars['USE_NCCL']:
            report('-- Building NCCL library')
        else:
            report('-- Not using NCCL')
        if cmake_cache_vars['USE_DISTRIBUTED']:
            if IS_WINDOWS:
                report('-- Building without distributed package')
            else:
                report('-- Building with distributed package: ')
                report('  -- USE_TENSORPIPE={}'.format(cmake_cache_vars['USE_TENSORPIPE']))
                report('  -- USE_GLOO={}'.format(cmake_cache_vars['USE_GLOO']))
                report('  -- USE_MPI={}'.format(cmake_cache_vars['USE_OPENMPI']))
        else:
            report('-- Building without distributed package')
        if cmake_cache_vars['STATIC_DISPATCH_BACKEND']:
            report('-- Using static dispatch with backend {}'.format(cmake_cache_vars['STATIC_DISPATCH_BACKEND']))
        if cmake_cache_vars['USE_LIGHTWEIGHT_DISPATCH']:
            report('-- Using lightweight dispatch')
        if cmake_cache_vars['BUILD_EXECUTORCH']:
            report('-- Building Executorch')

        if cmake_cache_vars['USE_ITT']:
            report('-- Using ITT')
        else:
            report('-- Not using ITT')

        if cmake_cache_vars['BUILD_NVFUSER']:
            report('-- Building nvfuser')
        else:
            report('-- Not Building nvfuser')

        # Do not use clang to compile extensions if `-fstack-clash-protection` is defined
        # in system CFLAGS
        c_flags = str(os.getenv('CFLAGS', ''))
        if IS_LINUX and '-fstack-clash-protection' in c_flags and 'clang' in os.environ.get('CC', ''):
            os.environ['CC'] = str(os.environ['CC'])

        # It's an old-style class in Python 2.7...
        setuptools.command.build_ext.build_ext.run(self)

        if IS_DARWIN and package_type != 'conda':
            self._embed_libiomp()

        # Copy the essential export library to compile C++ extensions.
        if IS_WINDOWS:
            build_temp = self.build_temp

            ext_filename = self.get_ext_filename('_C')
            lib_filename = '.'.join(ext_filename.split('.')[:-1]) + '.lib'

            export_lib = os.path.join(
                build_temp, 'torch', 'csrc', lib_filename).replace('\\', '/')

            build_lib = self.build_lib

            target_lib = os.path.join(
                build_lib, 'torch', 'lib', '_C.lib').replace('\\', '/')

            # Create "torch/lib" directory if not exists.
            # (It is not created yet in "develop" mode.)
            target_dir = os.path.dirname(target_lib)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            self.copy_file(export_lib, target_lib)

    def build_extensions(self):
        self.create_compile_commands()
        # The caffe2 extensions are created in
        # tmp_install/lib/pythonM.m/site-packages/caffe2/python/
        # and need to be copied to build/lib.linux.... , which will be a
        # platform dependent build folder created by the "build" command of
        # setuptools. Only the contents of this folder are installed in the
        # "install" command by default.
        # We only make this copy for Caffe2's pybind extensions
        caffe2_pybind_exts = [
            'caffe2.python.caffe2_pybind11_state',
            'caffe2.python.caffe2_pybind11_state_gpu',
            'caffe2.python.caffe2_pybind11_state_hip',
        ]
        i = 0
        while i < len(self.extensions):
            ext = self.extensions[i]
            if ext.name not in caffe2_pybind_exts:
                i += 1
                continue
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            report("\nCopying extension {}".format(ext.name))

            relative_site_packages = sysconfig.get_path('purelib').replace(sysconfig.get_path('data'), '').lstrip(os.path.sep)
            src = os.path.join("torch", relative_site_packages, filename)
            if not os.path.exists(src):
                report("{} does not exist".format(src))
                del self.extensions[i]
            else:
                dst = os.path.join(os.path.realpath(self.build_lib), filename)
                report("Copying {} from {} to {}".format(ext.name, src, dst))
                dst_dir = os.path.dirname(dst)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                self.copy_file(src, dst)
                i += 1

        # Copy functorch extension
        for i, ext in enumerate(self.extensions):
            if ext.name != "functorch._C":
                continue
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            fileext = os.path.splitext(filename)[1]
            src = os.path.join(os.path.dirname(filename), "functorch" + fileext)
            dst = os.path.join(os.path.realpath(self.build_lib), filename)
            if os.path.exists(src):
                report("Copying {} from {} to {}".format(ext.name, src, dst))
                dst_dir = os.path.dirname(dst)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                self.copy_file(src, dst)

        # Copy nvfuser extension
        for i, ext in enumerate(self.extensions):
            if ext.name != "nvfuser._C":
                continue
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            fileext = os.path.splitext(filename)[1]
            src = os.path.join(os.path.dirname(filename), "nvfuser" + fileext)
            dst = os.path.join(os.path.realpath(self.build_lib), filename)
            if os.path.exists(src):
                report("Copying {} from {} to {}".format(ext.name, src, dst))
                dst_dir = os.path.dirname(dst)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                self.copy_file(src, dst)

        setuptools.command.build_ext.build_ext.build_extensions(self)


    def get_outputs(self):
        outputs = setuptools.command.build_ext.build_ext.get_outputs(self)
        outputs.append(os.path.join(self.build_lib, "caffe2"))
        report("setup.py::get_outputs returning {}".format(outputs))
        return outputs

    def create_compile_commands(self):
        def load(filename):
            with open(filename) as f:
                return json.load(f)
        ninja_files = glob.glob('build/*compile_commands.json')
        cmake_files = glob.glob('torch/lib/build/*/compile_commands.json')
        all_commands = [entry
                        for f in ninja_files + cmake_files
                        for entry in load(f)]

        # cquery does not like c++ compiles that start with gcc.
        # It forgets to include the c++ header directories.
        # We can work around this by replacing the gcc calls that python
        # setup.py generates with g++ calls instead
        for command in all_commands:
            if command['command'].startswith("gcc "):
                command['command'] = "g++ " + command['command'][4:]

        new_contents = json.dumps(all_commands, indent=2)
        contents = ''
        if os.path.exists('compile_commands.json'):
            with open('compile_commands.json', 'r') as f:
                contents = f.read()
        if contents != new_contents:
            with open('compile_commands.json', 'w') as f:
                f.write(new_contents)

class concat_license_files():
    """Merge LICENSE and LICENSES_BUNDLED.txt as a context manager

    LICENSE is the main PyTorch license, LICENSES_BUNDLED.txt is auto-generated
    from all the licenses found in ./third_party/. We concatenate them so there
    is a single license file in the sdist and wheels with all of the necessary
    licensing info.
    """
    def __init__(self, include_files=False):
        self.f1 = 'LICENSE'
        self.f2 = 'third_party/LICENSES_BUNDLED.txt'
        self.include_files = include_files

    def __enter__(self):
        """Concatenate files"""

        old_path = sys.path
        sys.path.append(third_party_path)
        try:
            from build_bundled import create_bundled
        finally:
            sys.path = old_path

        with open(self.f1, 'r') as f1:
            self.bsd_text = f1.read()

        with open(self.f1, 'a') as f1:
            f1.write('\n\n')
            create_bundled(os.path.relpath(third_party_path), f1,
                           include_files=self.include_files)


    def __exit__(self, exception_type, exception_value, traceback):
        """Restore content of f1"""
        with open(self.f1, 'w') as f:
            f.write(self.bsd_text)


try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    # This is useful when wheel is not installed and bdist_wheel is not
    # specified on the command line. If it _is_ specified, parsing the command
    # line will fail before wheel_concatenate is needed
    wheel_concatenate = None
else:
    # Need to create the proper LICENSE.txt for the wheel
    class wheel_concatenate(bdist_wheel):
        """ check submodules on sdist to prevent incomplete tarballs """
        def run(self):
            with concat_license_files(include_files=True):
                super().run()


class install(setuptools.command.install.install):
    def run(self):
        super().run()


class clean(setuptools.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import glob
        import re
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
            for wildcard in filter(None, ignores.split('\n')):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    # Don't remove absolute paths from the system
                    wildcard = wildcard.lstrip('./')

                    for filename in glob.glob(wildcard):
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)


class sdist(setuptools.command.sdist.sdist):
    def run(self):
        with concat_license_files():
            super().run()


def get_cmake_cache_vars():
    try:
        return defaultdict(lambda: False, cmake.get_cmake_cache_variables())
    except FileNotFoundError:
        # CMakeCache.txt does not exist. Probably running "python setup.py clean" over a clean directory.
        return defaultdict(lambda: False)

#这一函数内说明了c/c++文件在配置阶段是如何与torch项目相关联的
def configure_extension_build():
    r"""Configures extension build options according to system environment and user's choice.

    Returns:
      The input to parameters ext_modules, cmdclass, packages, and entry_points as required in setuptools.setup.
    """

    cmake_cache_vars = get_cmake_cache_vars()

    ################################################################################
    # Configure compile flags
    ################################################################################

    library_dirs = []
    extra_install_requires = []

    if IS_WINDOWS:
        # /NODEFAULTLIB makes sure we only link to DLL runtime
        # and matches the flags set for protobuf and ONNX
        extra_link_args = ['/NODEFAULTLIB:LIBCMT.LIB']
        # /MD links against DLL runtime
        # and matches the flags set for protobuf and ONNX
        # /EHsc is about standard C++ exception handling
        # /DNOMINMAX removes builtin min/max functions
        # /wdXXXX disables warning no. XXXX
        extra_compile_args = ['/MD', '/FS', '/EHsc', '/DNOMINMAX',
                              '/wd4267', '/wd4251', '/wd4522', '/wd4522', '/wd4838',
                              '/wd4305', '/wd4244', '/wd4190', '/wd4101', '/wd4996',
                              '/wd4275']
    else:
        extra_link_args = []
        extra_compile_args = [
            '-Wall',
            '-Wextra',
            '-Wno-strict-overflow',
            '-Wno-unused-parameter',
            '-Wno-missing-field-initializers',
            '-Wno-write-strings',
            '-Wno-unknown-pragmas',
            # This is required for Python 2 declarations that are deprecated in 3.
            '-Wno-deprecated-declarations',
            # Python 2.6 requires -fno-strict-aliasing, see
            # http://legacy.python.org/dev/peps/pep-3123/
            # We also depend on it in our code (even Python 3).
            '-fno-strict-aliasing',
            # Clang has an unfixed bug leading to spurious missing
            # braces warnings, see
            # https://bugs.llvm.org/show_bug.cgi?id=21629
            '-Wno-missing-braces',
        ]

    library_dirs.append(lib_path)

    main_compile_args = []
    main_libraries = ['torch_python']
    main_link_args = []

    '''
    之前提到的configure_extension_build()函数中与c/c++相关的片段，
    注意到在声明Extension类时，其中提到过sources参数，
    而其所传递的main_sources变量在上文被定义过，即：
    '''
    main_sources = ["torch/csrc/stub.c"]

    if cmake_cache_vars['USE_CUDA']:
        library_dirs.append(
            os.path.dirname(cmake_cache_vars['CUDA_CUDA_LIB']))

    if build_type.is_debug():
        if IS_WINDOWS:
            extra_compile_args.append('/Z7')
            extra_link_args.append('/DEBUG:FULL')
        else:
            extra_compile_args += ['-O0', '-g']
            extra_link_args += ['-O0', '-g']

    if build_type.is_rel_with_deb_info():
        if IS_WINDOWS:
            extra_compile_args.append('/Z7')
            extra_link_args.append('/DEBUG:FULL')
        else:
            extra_compile_args += ['-g']
            extra_link_args += ['-g']

    # special CUDA 11.7 package that requires installation of cuda runtime, cudnn and cublas
    pytorch_extra_install_requirements = os.getenv("PYTORCH_EXTRA_INSTALL_REQUIREMENTS", "")
    if pytorch_extra_install_requirements:
        report(f"pytorch_extra_install_requirements: {pytorch_extra_install_requirements}")
        extra_install_requires += pytorch_extra_install_requirements.split("|")


    # Cross-compile for M1
    if IS_DARWIN:
        macos_target_arch = os.getenv('CMAKE_OSX_ARCHITECTURES', '')
        if macos_target_arch in ['arm64', 'x86_64']:
            macos_sysroot_path = os.getenv('CMAKE_OSX_SYSROOT')
            if macos_sysroot_path is None:
                macos_sysroot_path = subprocess.check_output([
                    'xcrun', '--show-sdk-path', '--sdk', 'macosx'
                ]).decode('utf-8').strip()
            extra_compile_args += ['-arch', macos_target_arch, '-isysroot', macos_sysroot_path]
            extra_link_args += ['-arch', macos_target_arch]


    def make_relative_rpath_args(path):
        if IS_DARWIN:
            return ['-Wl,-rpath,@loader_path/' + path]
        elif IS_WINDOWS:
            return []
        else:
            return ['-Wl,-rpath,$ORIGIN/' + path]

    ################################################################################
    # Declare extensions and package
    ################################################################################

    '''
    pytorch项目是在此处将c/c++语言作为Extension纳入自己的项目，
    并预备在后文的setup()进行编译的。
    注意此处的Extension类也为setuptools定义好的一个类，
    具体的作用将实例化时传入的各种参数包装好，留待setup()时一并使用。
    '''
    extensions = []
    excludes = ['tools', 'tools.*']
    if not cmake_cache_vars['BUILD_CAFFE2']:
        excludes.extend(['caffe2', 'caffe2.*'])
    if not cmake_cache_vars['BUILD_FUNCTORCH']:
        excludes.extend(['functorch', 'functorch.*'])
    if not cmake_cache_vars['BUILD_NVFUSER']:
        excludes.extend(['nvfuser', 'nvfuser.*'])
    packages = find_packages(exclude=excludes)
    C = Extension("torch._C",
                  libraries=main_libraries,
                  sources=main_sources,
                  language='c',
                  extra_compile_args=main_compile_args + extra_compile_args,
                  include_dirs=[],
                  library_dirs=library_dirs,
                  extra_link_args=extra_link_args + main_link_args + make_relative_rpath_args('lib'))
    C_flatbuffer = Extension("torch._C_flatbuffer",
                             libraries=main_libraries,
                             sources=["torch/csrc/stub_with_flatbuffer.c"],
                             language='c',
                             extra_compile_args=main_compile_args + extra_compile_args,
                             include_dirs=[],
                             library_dirs=library_dirs,
                             extra_link_args=extra_link_args + main_link_args + make_relative_rpath_args('lib'))
    extensions.append(C)
    extensions.append(C_flatbuffer)

    # These extensions are built by cmake and copied manually in build_extensions()
    # inside the build_ext implementation
    if cmake_cache_vars['USE_ROCM']:
        triton_req_file = os.path.join(cwd, ".github", "requirements", "triton-requirements-rocm.txt")
        if os.path.exists(triton_req_file):
            with open(triton_req_file) as f:
                triton_req = f.read().strip()
                extra_install_requires.append(triton_req)

    if cmake_cache_vars['BUILD_CAFFE2']:
        extensions.append(
            Extension(
                name=str('caffe2.python.caffe2_pybind11_state'),
                sources=[]),
        )
        if cmake_cache_vars['USE_CUDA']:
            extensions.append(
                Extension(
                    name=str('caffe2.python.caffe2_pybind11_state_gpu'),
                    sources=[]),
            )
        if cmake_cache_vars['USE_ROCM']:
            extensions.append(
                Extension(
                    name=str('caffe2.python.caffe2_pybind11_state_hip'),
                    sources=[]),
            )
    if cmake_cache_vars['BUILD_FUNCTORCH']:
        extensions.append(
            Extension(
                name=str('functorch._C'),
                sources=[]),
        )
    if cmake_cache_vars['BUILD_NVFUSER']:
        extensions.append(
            Extension(
                name=str('nvfuser._C'),
                sources=[]),
        )

    cmdclass = {
        'bdist_wheel': wheel_concatenate,
        'build_ext': build_ext,
        'clean': clean,
        'install': install,
        'sdist': sdist,
    }

    entry_points = {
        'console_scripts': [
            'convert-caffe2-to-onnx = caffe2.python.onnx.bin.conversion:caffe2_to_onnx',
            'convert-onnx-to-caffe2 = caffe2.python.onnx.bin.conversion:onnx_to_caffe2',
            'torchrun = torch.distributed.run:main',
        ]
    }

    return extensions, cmdclass, packages, entry_points, extra_install_requires

# post run, warnings, printed at the end to make them more visible
build_update_message = """
    It is no longer necessary to use the 'build' or 'rebuild' targets

    To install:
      $ python setup.py install
    To develop locally:
      $ python setup.py develop
    To force cmake to re-generate native build files (off by default):
      $ python setup.py develop --cmake
"""


def print_box(msg):
    lines = msg.split('\n')
    size = max(len(l) + 1 for l in lines)
    print('-' * (size + 2))
    for l in lines:
        print('|{}{}|'.format(l, ' ' * (size - len(l))))
    print('-' * (size + 2))


def main():
    # the list of runtime dependencies required by this built package
    install_requires = [
        'filelock',
        'typing-extensions',
        'sympy',
        'networkx',
        'jinja2',
    ]

    extras_require = {
        'opt-einsum': ['opt-einsum>=3.3']
    }

    # Parse the command line and check the arguments before we proceed with
    # building deps and setup. We need to set values so `--help` works.

    #Distribution()类是定义在setuptools中的一个类，其作用就是检查主机环境是否含有项目所要求的所有的依赖，若没有则自动安装。
    dist = Distribution()
    dist.script_name = os.path.basename(sys.argv[0])
    dist.script_args = sys.argv[1:]
    try:
        dist.parse_command_line()
    except setuptools.distutils.errors.DistutilsArgError as e:
        print(e)
        sys.exit(1)

    #检查项目的一些临时文件是否已经被创建，若没有则创建目录和文件，保证本机环境和torch需要的安装环境相同。
    mirror_files_into_torchgen()

    if RUN_BUILD_DEPS:
        build_deps()

    extensions, cmdclass, packages, entry_points, extra_install_requires = configure_extension_build()

    install_requires += extra_install_requires

    # Read in README.md for our long_description
    with open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

    version_range_max = max(sys.version_info[1], 10) + 1
    torch_package_data = [
        'py.typed',
        'bin/*',
        'test/*',
        '_C/*.pyi',
        '_C_flatbuffer/*.pyi',
        'cuda/*.pyi',
        'optim/*.pyi',
        'autograd/*.pyi',
        'nn/*.pyi',
        'nn/modules/*.pyi',
        'nn/parallel/*.pyi',
        'utils/data/*.pyi',
        'utils/data/datapipes/*.pyi',
        'lib/*.so*',
        'lib/*.dylib*',
        'lib/*.dll',
        'lib/*.lib',
        'lib/*.pdb',
        'lib/torch_shm_manager',
        'lib/*.h',
        'include/*.h',
        'include/ATen/*.h',
        'include/ATen/cpu/*.h',
        'include/ATen/cpu/vec/vec256/*.h',
        'include/ATen/cpu/vec/vec256/vsx/*.h',
        'include/ATen/cpu/vec/vec512/*.h',
        'include/ATen/cpu/vec/*.h',
        'include/ATen/core/*.h',
        'include/ATen/cuda/*.cuh',
        'include/ATen/cuda/*.h',
        'include/ATen/cuda/detail/*.cuh',
        'include/ATen/cuda/detail/*.h',
        'include/ATen/cudnn/*.h',
        'include/ATen/functorch/*.h',
        'include/ATen/ops/*.h',
        'include/ATen/hip/*.cuh',
        'include/ATen/hip/*.h',
        'include/ATen/hip/detail/*.cuh',
        'include/ATen/hip/detail/*.h',
        'include/ATen/hip/impl/*.h',
        'include/ATen/miopen/*.h',
        'include/ATen/detail/*.h',
        'include/ATen/native/*.h',
        'include/ATen/native/cpu/*.h',
        'include/ATen/native/cuda/*.h',
        'include/ATen/native/cuda/*.cuh',
        'include/ATen/native/hip/*.h',
        'include/ATen/native/hip/*.cuh',
        'include/ATen/native/quantized/*.h',
        'include/ATen/native/quantized/cpu/*.h',
        'include/ATen/quantized/*.h',
        'include/caffe2/serialize/*.h',
        'include/c10/*.h',
        'include/c10/macros/*.h',
        'include/c10/core/*.h',
        'include/ATen/core/boxing/*.h',
        'include/ATen/core/boxing/impl/*.h',
        'include/ATen/core/dispatch/*.h',
        'include/ATen/core/op_registration/*.h',
        'include/c10/core/impl/*.h',
        'include/c10/util/*.h',
        'include/c10/cuda/*.h',
        'include/c10/cuda/impl/*.h',
        'include/c10/hip/*.h',
        'include/c10/hip/impl/*.h',
        'include/torch/*.h',
        'include/torch/csrc/*.h',
        'include/torch/csrc/api/include/torch/*.h',
        'include/torch/csrc/api/include/torch/data/*.h',
        'include/torch/csrc/api/include/torch/data/dataloader/*.h',
        'include/torch/csrc/api/include/torch/data/datasets/*.h',
        'include/torch/csrc/api/include/torch/data/detail/*.h',
        'include/torch/csrc/api/include/torch/data/samplers/*.h',
        'include/torch/csrc/api/include/torch/data/transforms/*.h',
        'include/torch/csrc/api/include/torch/detail/*.h',
        'include/torch/csrc/api/include/torch/detail/ordered_dict.h',
        'include/torch/csrc/api/include/torch/nn/*.h',
        'include/torch/csrc/api/include/torch/nn/functional/*.h',
        'include/torch/csrc/api/include/torch/nn/options/*.h',
        'include/torch/csrc/api/include/torch/nn/modules/*.h',
        'include/torch/csrc/api/include/torch/nn/modules/container/*.h',
        'include/torch/csrc/api/include/torch/nn/parallel/*.h',
        'include/torch/csrc/api/include/torch/nn/utils/*.h',
        'include/torch/csrc/api/include/torch/optim/*.h',
        'include/torch/csrc/api/include/torch/optim/schedulers/*.h',
        'include/torch/csrc/api/include/torch/serialize/*.h',
        'include/torch/csrc/autograd/*.h',
        'include/torch/csrc/autograd/functions/*.h',
        'include/torch/csrc/autograd/generated/*.h',
        'include/torch/csrc/autograd/utils/*.h',
        'include/torch/csrc/cuda/*.h',
        'include/torch/csrc/distributed/c10d/*.h',
        'include/torch/csrc/distributed/c10d/*.hpp',
        'include/torch/csrc/distributed/rpc/*.h',
        'include/torch/csrc/jit/*.h',
        'include/torch/csrc/jit/backends/*.h',
        'include/torch/csrc/jit/generated/*.h',
        'include/torch/csrc/jit/passes/*.h',
        'include/torch/csrc/jit/passes/quantization/*.h',
        'include/torch/csrc/jit/passes/utils/*.h',
        'include/torch/csrc/jit/runtime/*.h',
        'include/torch/csrc/jit/ir/*.h',
        'include/torch/csrc/jit/frontend/*.h',
        'include/torch/csrc/jit/api/*.h',
        'include/torch/csrc/jit/serialization/*.h',
        'include/torch/csrc/jit/python/*.h',
        'include/torch/csrc/jit/mobile/*.h',
        'include/torch/csrc/jit/testing/*.h',
        'include/torch/csrc/jit/tensorexpr/*.h',
        'include/torch/csrc/jit/tensorexpr/operators/*.h',
        'include/torch/csrc/jit/codegen/cuda/*.h',
        'include/torch/csrc/jit/codegen/cuda/ops/*.h',
        'include/torch/csrc/jit/codegen/cuda/scheduler/*.h',
        'include/torch/csrc/onnx/*.h',
        'include/torch/csrc/profiler/*.h',
        'include/torch/csrc/profiler/orchestration/*.h',
        'include/torch/csrc/profiler/stubs/*.h',
        'include/torch/csrc/utils/*.h',
        'include/torch/csrc/tensor/*.h',
        'include/torch/csrc/lazy/backend/*.h',
        'include/torch/csrc/lazy/core/*.h',
        'include/torch/csrc/lazy/core/internal_ops/*.h',
        'include/torch/csrc/lazy/core/ops/*.h',
        'include/torch/csrc/lazy/ts_backend/*.h',
        'include/pybind11/*.h',
        'include/pybind11/detail/*.h',
        'include/TH/*.h*',
        'include/TH/generic/*.h*',
        'include/THC/*.cuh',
        'include/THC/*.h*',
        'include/THC/generic/*.h',
        'include/THH/*.cuh',
        'include/THH/*.h*',
        'include/THH/generic/*.h',
        'include/sleef.h',
        "_inductor/codegen/*.h",
        'share/cmake/ATen/*.cmake',
        'share/cmake/Caffe2/*.cmake',
        'share/cmake/Caffe2/public/*.cmake',
        'share/cmake/Caffe2/Modules_CUDA_fix/*.cmake',
        'share/cmake/Caffe2/Modules_CUDA_fix/upstream/*.cmake',
        'share/cmake/Caffe2/Modules_CUDA_fix/upstream/FindCUDA/*.cmake',
        'share/cmake/Gloo/*.cmake',
        'share/cmake/Tensorpipe/*.cmake',
        'share/cmake/Torch/*.cmake',
        'utils/benchmark/utils/*.cpp',
        'utils/benchmark/utils/valgrind_wrapper/*.cpp',
        'utils/benchmark/utils/valgrind_wrapper/*.h',
        'utils/model_dump/skeleton.html',
        'utils/model_dump/code.js',
        'utils/model_dump/*.mjs',
    ]

    if get_cmake_cache_vars()['BUILD_CAFFE2']:
        torch_package_data.extend([
            'include/caffe2/**/*.h',
            'include/caffe2/utils/*.h',
            'include/caffe2/utils/**/*.h',
        ])
    torchgen_package_data = [
        # Recursive glob doesn't work in setup.py,
        # https://github.com/pypa/setuptools/issues/1806
        # To make this robust we should replace it with some code that
        # returns a list of everything under packaged/
        'packaged/ATen/*',
        'packaged/ATen/native/*',
        'packaged/ATen/templates/*',
    ]
    '''
    setup()函数无疑是setup.py()的核心，
    其本身也是由setuptools实现好的函数，
    pytorch项目将各种该函数需要的参数设计好，
    再传入setup()函数中，
    即可方便快捷的进行项目部署。
    '''

    '''
    4，cmdclass，setup的操作参数，是一个字典：

'create_version_file': <class '__main__.create_version_file'>
'build': <class '__main__.build'>
'build_py': <class '__main__.build_py'>
'build_ext': <class '__main__.build_ext'>
'build_deps': <class '__main__.build_deps'>
'build_module': <class '__main__.build_module'>
'rebuild': <class '__main__.rebuild'>
'develop': <class '__main__.develop'>
'install': <class '__main__.install'>
'clean': <class '__main__.clean'>
'build_caffe2': <class '__main__.build_dep'>, 
'rebuild_caffe2': <class '__main__.rebuild_dep'
因为我们执行的是python setup.py install，所以这里将会调用__main__.install（中的run方法）。install所做的工作就是调用build_deps和install。Gemfield将在下面的章节中探讨。

5，packages，指定项目中python源代码的路径。通常使用find_packages() 默认在和setup.py同一目录下搜索各个含有 __init__.py的包；

6，entry_points使用python的机制来注册你的命令，比如在PyTorch的setup.py中，entry_points如下所示：

entry_points = {
    'console_scripts': [
        'convert-caffe2-to-onnx = caffe2.python.onnx.bin.conversion:caffe2_to_onnx',
        'convert-onnx-to-caffe2 = caffe2.python.onnx.bin.conversion:onnx_to_caffe2',
    ]
}

以convert-caffe2-to-onnx为例，setup.py会在PATH路径下生成convert-caffe2-to-onnx文件，这个文件的内容如下：

#!/root/miniconda3/bin/python3
# EASY-INSTALL-ENTRY-SCRIPT: 'torch==1.1.0a0+ffd6138','console_scripts','convert-caffe2-to-onnx'
__requires__ = 'torch==1.1.0a0+ffd6138'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('torch==1.1.0a0+ffd6138', 'console_scripts', 'convert-caffe2-to-onnx')()
    )
7，package_data，指定将什么文件打包到package里，使用这个参数一般是因为有些要打包的文件是动态生成的（也就是执行setup.py过程中生成的）。
在PyTorch里，package_data主要是一些动态库和头文件。

yk====很多东西可能有变化， 但是大体思路还对！！！



使用cmake生成Makefile
使用tools/build_pytorch_libs.sh脚本来编译。在默认情况下，PyTorch会使用下面的命令和环境变量来开始caffe2的编译：
。。。。


这个命令会创建torch/lib/tmp_install目录，接着使用cmake工具生成Makefile：

cmake /home/gemfield/github/pytorch -DPYTHON_EXECUTABLE=/root/miniconda3/bin/python -DPYTHON_LIBRARY=/root/miniconda3/lib/libpython3.7m.so.1.0 -DPYTHON_INCLUDE_DIR=/root/miniconda3/include/python3.7m -DBUILDING_WITH_TORCH_LIBS=ON -DTORCH_BUILD_VERSION=1.1.0a0+ffd6138 -DCMAKE_BUILD_TYPE=Release -DBUILD_TORCH=ON -DBUILD_PYTHON=ON -DBUILD_SHARED_LIBS=ON -DBUILD_BINARY=OFF -DBUILD_TEST=ON -DINSTALL_TEST=ON -DBUILD_CAFFE2_OPS=ON -DONNX_NAMESPACE=onnx_torch -DUSE_CUDA=1 -DUSE_DISTRIBUTED=ON -DUSE_FBGEMM=1 -DUSE_NUMPY= -DNUMPY_INCLUDE_DIR= -DUSE_SYSTEM_NCCL=ON -DNCCL_INCLUDE_DIR=/usr/include -DNCCL_ROOT_DIR=/usr/ -DNCCL_SYSTEM_LIB=/usr/lib/x86_64-linux-gnu/libnccl.so.2.3.7 -DCAFFE2_STATIC_LINK_CUDA=0 -DUSE_ROCM=0 -DUSE_NNPACK=1 -DUSE_LEVELDB=OFF -DUSE_LMDB=OFF -DUSE_OPENCV=OFF -DUSE_QNNPACK=1 -DUSE_TENSORRT=OFF -DUSE_FFMPEG=OFF -DUSE_SYSTEM_EIGEN_INSTALL=OFF -DCUDNN_INCLUDE_DIR=/usr/include/ -DCUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/ -DCUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so.7 -DUSE_MKLDNN=1 -DNCCL_EXTERNAL=1 -DCMAKE_INSTALL_PREFIX=/home/gemfield/github/pytorch/torch/lib/tmp_install -DCMAKE_C_FLAGS= -DCMAKE_CXX_FLAGS= '-DCMAKE_EXE_LINKER_FLAGS= -Wl,-rpath,$ORIGIN ' '-DCMAKE_SHARED_LINKER_FLAGS= -Wl,-rpath,$ORIGIN ' -DTHD_SO_VERSION=1 -DCMAKE_PREFIX_PATH=/root/miniconda3/lib/python3.7/site-packages
cmake的关键configure信息如下（Makefile等配置信息会写到build目录下）：

-- General:
--   BLAS                  : MKL
--   CXX flags             :  -Wno-deprecated -fvisibility-inlines-hidden -fopenmp -DUSE_FBGEMM -O2 -fPIC -Wno-narrowing -Wall -Wextra -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -Wno-unused-but-set-variable -Wno-maybe-uninitialized
--   Compile definitions   : TH_BLAS_MKL;ONNX_NAMESPACE=onnx_torch;USE_GCC_ATOMICS=1;HAVE_MMAP=1;_FILE_OFFSET_BITS=64;HAVE_SHM_OPEN=1;HAVE_SHM_UNLINK=1;HAVE_MALLOC_USABLE_SIZE=1
--   CMAKE_PREFIX_PATH     : /root/miniconda3/lib/python3.7/site-packages
......
--   Public Dependencies  : Threads::Threads;caffe2::mkl;caffe2::mkldnn
--   Private Dependencies : qnnpack;nnpack;cpuinfo;fbgemm;fp16;gloo;aten_op_header_gen;onnxifi_loader;rt;gcc_s;gcc;dl


Make命令——中间源文件的产生
cmake生成Makefile后，编译系统接着调用make命令来进行编译和install：

make install -j24
这个编译过程一共产生两种文件（Linux上）：

1，根据模板生成源文件，包含cpp文件和Python 文件（大部分是UT）；

2，生成.o/.a文件 、动态库 .so文件、可执行文件。

1，ATen部分cpp文件：

2，torch部分cpp文件：

3，python文件部分：


Make命令——编译三方库
在开始阶段会编译third_party目录下的依赖包（基本是facebook和谷歌公司贡献的）。

Make命令——生成静态库、动态库、可执行文件
编译过程中生成的.a文件：

编译过程生成的可执行文件：


编译过程中生成的.so动态库：

其中：

1，lib/libc10.so由下列源文件编译生成：

c10/*.cpp
2，lib/libc10_cuda.so由下列源文件编译生成：

c10/cuda/CUDAStream.cpp
c10/cuda/impl/CUDAGuardImpl.cpp
c10/cuda/impl/CUDATest.cpp
3，lib/libshm.so由下列源文件编译生成：

torch/lib/libshm/core.cpp
4，lib/libcaffe2.so由下列源文件编译生成：

aten/*.cpp
caffe2/core/*.cpp

...
5，lib/libcaffe2_gpu.so由下列源文件编译生成：

aten/src/TH/*
aten/src/THCUNN/*

...
6，caffe2_pybind11_state.cpython-37m-x86_64-linux-gnu.so由下列源文件编译生成：

caffe2/python/pybind_state.cc
caffe2/python/pybind_state_dlpack.cc

7，caffe2_pybind11_state_gpu.cpython-37m-x86_64-linux-gnu.so由下列源文件编译生成：

caffe2/python/pybind_state.cc
caffe2/python/pybind_state_dlpack.cc

8，lib/libtorch.so由下列源文件编译生成：

torch/csrc/autograd/*.cpp
torch/csrc/autograd/generated/*.cpp

9，lib/libtorch_python.so由下列源文件编译生成：

torch/lib/THD/*.cpp
torch/lib/c10d/*.cpp

注意libtorch_python.so编译的时候也需要链接libtorch.so文件。

Make命令——拷贝文件到合适路径下
这一步的核心工作就是使用make install来将make过程中产生的文件：

pytorch/torch/lib/tmp_install/lib/libtorch.so.1 
pytorch/torch/lib/tmp_install/lib/libtorch_python.so 
pytorch/torch/lib/tmp_install/lib/pkgconfig 
pytorch/torch/lib/tmp_install/lib/python3.7
拷贝到pytorch/torch/lib目录下。使用的是rsync命令：

rsync -lptgoD -r /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libTHD.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libasmjit.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libc10.so /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libc10_cuda.so /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libc10d.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libcaffe2.so /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libcaffe2_detectron_ops_gpu.so /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libcaffe2_gpu.so /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libcaffe2_module_test_dynamic.so /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libcaffe2_observers.so /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libclog.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libcpuinfo.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libfbgemm.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libgloo.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libgloo_builder.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libgloo_cuda.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libmkldnn.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libnnpack.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libonnx.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libonnx_proto.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libonnxifi.so /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libonnxifi_dummy.so /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libonnxifi_loader.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libprotobuf-lite.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libprotobuf.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libprotoc.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libpthreadpool.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libqnnpack.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libshm.so /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libsleef.a /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libtorch.so /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libtorch.so.1 /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/libtorch_python.so /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/pkgconfig /home/gemfield/github/pytorch/torch/lib/tmp_install/lib/python3.7 .
setuptools之build_py
这个步骤就是来生成python的package的Python部分，在这个阶段，我们要做的工作主要就是拷贝文件到符合package目录规范的文件夹里：

1，拷贝一些cpp的header文件到build/lib.linux-x86_64-3.7；

2，拷贝一些py文件到build/lib.linux-x86_64-3.7；

3，拷贝一些可执行文件到build/lib.linux-x86_64-3.7；

4，拷贝一些cuda header文件到build/lib.linux-x86_64-3.7；

5，拷贝一些cmake文件到build/lib.linux-x86_64-3.7；

6，拷贝一些zip文件到build/lib.linux-x86_64-3.7；

7，拷贝一些so文件到build/lib.linux-x86_64-3.7：

动态库so文件比较重要，在build_py阶段拷贝到build/lib.linux-x86_64-3.7的so文件有：

torch/lib/libc10_cuda.so -> build/lib.linux-x86_64-3.7/torch/lib
torch/lib/libcaffe2_detectron_ops_gpu.so -> build/lib.linux-x86_64-3.7/torch/lib
setuptools之build_ext
这个步骤是用来编译python的c/c++扩展插件的。先拷贝两个共享库到合适的目录下：

1，拷贝caffe2_pybind11_state.cpython-37m-x86_64-linux-gnu.so 到 build/lib.linux-x86_64-3.7/caffe2/python/caffe2_pybind11_state.cpython-37m-x86_64-linux-gnu.so（位于build/lib.linux-x86_64-3.7/caffe2/目录下）；


2，拷贝caffe2_pybind11_state_gpu.cpython-37m-x86_64-linux-gnu.so到build/lib.linux-x86_64-3.7/caffe2/python/caffe2_pybind11_state_gpu.cpython-37m-x86_64-linux-gnu.so（位于build/lib.linux-x86_64-3.7/caffe2/目录下）。

编译torch._C模块：

使用c++11标准和_THP_CORE 、ONNX_NAMESPACE=onnx_torch宏，用torch/csrc/stub.cpp源文件，链接libshm.so、libtorch_python.so、libcaffe2_gpu.so生成_C.cpython-37m-x86_64-linux-gnu.so 扩展插件（位于build/lib.linux-x86_64-3.7/torch/目录下）：

gcc -pthread -B /root/miniconda3/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/root/miniconda3/include/python3.7m -c torch/csrc/stub.cpp -o build/temp.linux-x86_64-3.7/torch/csrc/stub.o -D_THP_CORE -DONNX_NAMESPACE=onnx_torch -std=c++11 -Wall -Wextra -Wno-strict-overflow -Wno-unused-parameter -Wno-missing-field-initializers -Wno-write-strings -Wno-unknown-pragmas -Wno-deprecated-declarations -fno-strict-aliasing -Wno-missing-braces
g++ -pthread -shared -B /root/miniconda3/compiler_compat -L/root/miniconda3/lib -Wl,-rpath=/root/miniconda3/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.7/torch/csrc/stub.o -L/home/gemfield/github/pytorch/torch/lib -L/usr/local/cuda/lib64 -lshm -ltorch_python -o build/lib.linux-x86_64-3.7/torch/_C.cpython-37m-x86_64-linux-gnu.so -Wl,--no-as-needed /home/gemfield/github/pytorch/torch/lib/libcaffe2_gpu.so -Wl,--as-needed -Wl,-rpath,$ORIGIN/lib


编译torch._dl模块：

使用torch/csrc/dl.c编译出_dl.cpython-37m-x86_64-linux-gnu.so扩展插件（位于build/lib.linux-x86_64-3.7/torch/目录下）：

gcc -pthread -B /root/miniconda3/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/root/miniconda3/include/python3.7m -c torch/csrc/dl.c -o build/temp.linux-x86_64-3.7/torch/csrc/dl.o
gcc -pthread -shared -B /root/miniconda3/compiler_compat -L/root/miniconda3/lib -Wl,-rpath=/root/miniconda3/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.7/torch/csrc/dl.o -o build/lib.linux-x86_64-3.7/torch/_dl.cpython-37m-x86_64-linux-gnu.so


编译torch._nvrtc模块：

使用c++11标准，用torch/csrc/nvrtc.cpp源文件和_THP_CORE、ONNX_NAMESPACE=onnx_torch宏，并且链接libcuda.so、 -libnvrtc.so库，从而生成_nvrtc.cpython-37m-x86_64-linux-gnu.so扩展插件（位于build/lib.linux-x86_64-3.7/torch/目录下）：

gcc -pthread -B /root/miniconda3/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/home/gemfield/github/pytorch -I/root/miniconda3/include/python3.7m -c torch/csrc/nvrtc.cpp -o build/temp.linux-x86_64-3.7/torch/csrc/nvrtc.o -D_THP_CORE -DONNX_NAMESPACE=onnx_torch -std=c++11 -Wall -Wextra -Wno-strict-overflow -Wno-unused-parameter -Wno-missing-field-initializers -Wno-write-strings -Wno-unknown-pragmas -Wno-deprecated-declarations -fno-strict-aliasing -Wno-missing-braces
g++ -pthread -shared -B /root/miniconda3/compiler_compat -L/root/miniconda3/lib -Wl,-rpath=/root/miniconda3/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.7/torch/csrc/nvrtc.o -L/home/gemfield/github/pytorch/torch/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -o build/lib.linux-x86_64-3.7/torch/_nvrtc.cpython-37m-x86_64-linux-gnu.so -Wl,-rpath,$ORIGIN/lib -Wl,--no-as-needed -lcuda -lnvrtc


setuptools之install_lib
这一步完成了安装。

1，拷贝共享库到python lib路径下：

build/lib.linux-x86_64-3.7/caffe2/python/caffe2_pybind11_state_gpu.cpython-37m-x86_64-linux-gnu.so -> */site-packages/caffe2/python
build/lib.linux-x86_64-3.7/caffe2/python/caffe2_pybind11_state.cpython-37m-x86_64-linux-gnu.so -> */site-packages/caffe2/python


2，拷贝header文件到python路径下；

3，拷贝py文件到python路径下；

4，安装egg_info，也就是torch.egg-info：

running install_egg_info
running egg_info
writing torch.egg-info/PKG-INFO
writing dependency_links to torch.egg-info/dependency_links.txt
writing entry points to torch.egg-info/entry_points.txt
writing top-level names to torch.egg-info/top_level.txt
reading manifest file 'torch.egg-info/SOURCES.txt'
writing manifest file 'torch.egg-info/SOURCES.txt'
removing '/root/miniconda3/lib/python3.7/site-packages/torch-1.1.0a0+ffd6138-py3.7.egg-info' (and everything under it)
Copying torch.egg-info to /root/miniconda3/lib/python3.7/site-packages/torch-1.1.0a0+ffd6138-py3.7.egg-info


5，注册entry_points：

Installing convert-caffe2-to-onnx script to /root/miniconda3/bin
Installing convert-onnx-to-caffe2 script to /root/miniconda3/bin


庆祝
这之后，你就可以愉快的import torch了。让我们再次满怀喜悦的回忆下，import的torch是怎么得来的：

import torch
1，Python会找到sys.path中的torch目录，然后找到其中的__init__.py；

2，在这个文件中，会import torch._C，在Python3.7环境中，会加载torch目录中的_C.cpython-37m-x86_64-linux-gnu.so 动态扩展库；

3，_C.cpython-37m-x86_64-linux-gnu.so 动态库是由源文件torch/csrc/stub.cpp源文件，并且链接libtorch.so、libshm.so、libtorch_python.so、libcaffe2.so、libcaffe2_gpu.so、libc10_cuda.so、libc10.so生成的；当然不止这些库，还链接其它的三方库，这里gemfield只列出了由PyTorch仓库中编译出来的库。

另外，你也可以使用下面的命令佐证下：

gemfield@skyweb:~# ldd ./build/lib.linux-x86_64-3.7/torch/_C.cpython-37m-x86_64-linux-gnu.so | grep -i pytorch
        libshm.so => /home/gemfield/github/pytorch/./build/lib.linux-x86_64-3.7/torch/lib/libshm.so (0x00007f3f848c0000)
        libtorch_python.so => /home/gemfield/github/pytorch/./build/lib.linux-x86_64-3.7/torch/lib/libtorch_python.so (0x00007f3f83e55000)
        libcaffe2_gpu.so => /home/gemfield/github/pytorch/./build/lib.linux-x86_64-3.7/torch/lib/libcaffe2_gpu.so (0x00007f3f7263b000)
        libcaffe2.so => /home/gemfield/github/pytorch/./build/lib.linux-x86_64-3.7/torch/lib/libcaffe2.so (0x00007f3f6f57d000)
        libc10.so => /home/gemfield/github/pytorch/./build/lib.linux-x86_64-3.7/torch/lib/libc10.so (0x00007f3f6f34e000)
        libtorch.so.1 => /home/gemfield/github/pytorch/./build/lib.linux-x86_64-3.7/torch/lib/libtorch.so.1 (0x00007f3f6df54000)
        libc10_cuda.so => /home/gemfield/github/pytorch/./build/lib.linux-x86_64-3.7/torch/lib/libc10_cuda.so (0x00007f3f68d61000)


4，而这里列出来的libtorch.so、libshm.so、libtorch_python.so、libcaffe2.so、libcaffe2_gpu.so、libc10_cuda.so、libc10.so，已经由Gemfield在前文说明过了。
    '''
    setup(
        name=package_name,
        version=version,
        description=("Tensors and Dynamic neural networks in "
                     "Python with strong GPU acceleration"),
        long_description=long_description,
        long_description_content_type="text/markdown",
        ext_modules=extensions,   #这行代码将刚才我们提到的c/c++扩展包放入了安装过程中，
        cmdclass=cmdclass,
        packages=packages,
        entry_points=entry_points,
        install_requires=install_requires,
        extras_require=extras_require,
        package_data={
            'torch': torch_package_data,
            'torchgen': torchgen_package_data,
            'caffe2': [
                'python/serialized_test/data/operator_test/*.zip',
            ],
        },
        url='https://pytorch.org/',
        download_url='https://github.com/pytorch/pytorch/tags',
        author='PyTorch Team',
        author_email='packages@pytorch.org',
        python_requires='>={}'.format(python_min_version_str),
        # PyPI package information.
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Programming Language :: C++',
            'Programming Language :: Python :: 3',
        ] + ['Programming Language :: Python :: 3.{}'.format(i) for i in range(python_min_version[1], version_range_max)],
        license='BSD-3',
        keywords='pytorch, machine learning',
    )
    if EMIT_BUILD_WARNING:
        print_box(build_update_message)


if __name__ == '__main__':
    main()

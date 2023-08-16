#include <Python.h>

#ifdef _WIN32
__declspec(dllimport)
#endif
/*

接下来我们看看C++世界如何导出了TensorBase。
要在Python中能够import torch._C，则必须要使用Python的扩展规范来导出这个符号。
对于一个 Python module，共享库需要实现 PyInit_modulename 符号来作为import时候的逻辑入口。
对于PyTorch来说这个modulename 是_C。在torch/csrc/stub.cpp中 实现了PyInit__C这个函数。

未找到！
如果使用 JIT，则我们直接看 torch/csrc/deploy/interpreter/interpreter_impl.cpp，这里省略了众多代码。

struct ConcreteInterpreterImpl : public torch::deploy::InterpreterImpl {
  ConcreteInterpreterImpl() {
    PyImport_AppendInittab("torch._C", initModule);
}
*/
extern PyObject* initModule(void);

#ifndef _WIN32
#ifdef __cplusplus
extern "C"
#endif
__attribute__((visibility("default"))) PyObject* PyInit__C(void);
#endif

PyMODINIT_FUNC PyInit__C(void)
{
  return initModule();
}

/*
torch/csrc/stub.cpp链接libshm.so、libtorch_python.so、libcaffe2_gpu.so生成了_C.cpython-37m-x86_64-linux-gnu.so库，
而像前述方式import torch的时候，按照python规范，
会找到torch package目录下的__init__.py，在这个文件中进一步会调用：

from torch._C import *
其中torch._C就是_C.cpython-37m-x86_64-linux-gnu.so

按照Python规范，由于默认的引擎都是CPython，而CPython的C/C++扩展是一个共享库，并且这个共享库安装在PYTHONPATH目录下，
并且文件名（不包含后缀）要和module的名字一样，
并且这个共享库中要实现PyInit_modulename符号来作为import时候的逻辑入口。

对于PyTorch来说这个modulename 是_C，因此我们可以揣测，在torch/csrc/stub.cpp中一定实现了PyInit_C这个函数。是的，PyTorch就是这么做的，torch/csrc/stub.cpp中的代码就是下面这样：

extern PyObject* initModule(void);

PyMODINIT_FUNC PyInit__C(void)
{
  return initModule();
}
本文将从initModule函数展开，全面阐述PyTorch框架的初始化工作。
initModule就是PyTorch初始化时候的第一层调用栈了，因为所有的初始化工作都是在这个函数内完成的，内容比较多，gemfield将其划分为7部分：
*/
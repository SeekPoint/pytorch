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

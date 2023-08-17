#pragma once

#include <c10/core/DeviceType.h>
#include <c10/macros/Export.h>

#include <atomic>
#include <utility>

// Implements instruction set specific function dispatch.
//
// Kernels that may make use of specialized instruction sets (e.g. AVX2) are
// compiled multiple times with different compiler flags (e.g. -mavx2). A
// DispatchStub contains a table of function pointers for a kernel. At runtime,
// the fastest available kernel is chosen based on the features reported by
// cpuinfo.
//
// Example:
//
// In native/MyKernel.h:
//   using fn_type = void(*)(const Tensor& x);
//   DECLARE_DISPATCH(fn_type, stub);
//
// In native/MyKernel.cpp
//   DEFINE_DISPATCH(stub);
//
// In native/cpu/MyKernel.cpp:
//   namespace {
//     // use anonymous namespace so that different cpu versions won't conflict
//     void kernel(const Tensor& x) { ... }
//   }
//   REGISTER_DISPATCH(stub, &kernel);
//
// To call:
//   stub(kCPU, tensor);
//
// TODO: CPU instruction set selection should be folded into whatever
// the main dispatch mechanism is.

// ignore warnings about DispatchStub::DEFAULT, AVX, AVX2 defined elsewhere
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-var-template"
#endif

namespace at { namespace native {

enum class CPUCapability {
  DEFAULT = 0,
#if defined(HAVE_VSX_CPU_DEFINITION)
  VSX = 1,
#elif defined(HAVE_ZVECTOR_CPU_DEFINITION)
  ZVECTOR = 1,
#else
  AVX2 = 1,
  AVX512 = 2,
#endif
  NUM_OPTIONS
};

CPUCapability get_cpu_capability();

template <typename FnPtr, typename T>
struct DispatchStub;

/**
 * The sole purpose of this class is to outline methods that don't need to be
 * specialized or otherwise inlined and duplicated (by the compiler due to
 * template expansion), since it causes size bloat if there are a significant
 * number of specialization of the DispatchStub<> class.
 */
struct TORCH_API DispatchStubImpl {
  void* get_call_ptr(
    DeviceType device_type
    , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
      , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
      , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      , void *ZVECTOR
#endif
  );

  /**
   * The CPU Dispatch actual method is chosen in decreasing order of preference by
   * DispatchStubImpl::choose_cpu_impl() in case none is found by
   * DispatchStubImpl::get_call_ptr() in cpu_dispatch_ptr.
   */
  void* choose_cpu_impl(
    void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
    , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
    , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
    , void *ZVECTOR
#endif
  );

  // Fixing dispatch error in Windows debug builds.
  // See https://github.com/pytorch/pytorch/issues/22681 for more details.
  #if defined(_MSC_VER) && defined(_DEBUG)
    std::atomic<void*> cpu_dispatch_ptr;
    void* cuda_dispatch_ptr;
    void* hip_dispatch_ptr;
    void* mps_dispatch_ptr;
  #else
    std::atomic<void*> cpu_dispatch_ptr{nullptr};
    void* cuda_dispatch_ptr = nullptr;
    void* hip_dispatch_ptr = nullptr;
    void* mps_dispatch_ptr = nullptr;
  #endif
};

template <typename rT, typename T, typename... Args>
struct DispatchStub<rT (*)(Args...), T> {
  using FnPtr = rT (*) (Args...);

  DispatchStub() = default;
  DispatchStub(const DispatchStub&) = delete;
  DispatchStub& operator=(const DispatchStub&) = delete;

private:
  FnPtr get_call_ptr(DeviceType device_type) {
    return reinterpret_cast<FnPtr>(
      impl.get_call_ptr(device_type
      , reinterpret_cast<void*>(DEFAULT)
#ifdef HAVE_AVX512_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX512)
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX2)
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
      , reinterpret_cast<void*>(VSX)
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      , reinterpret_cast<void*>(ZVECTOR)
#endif
      )
    );
  }

public:
  template <typename... ArgTypes>
  rT operator()(DeviceType device_type, ArgTypes&&... args) {
    FnPtr call_ptr = get_call_ptr(device_type);
    return (*call_ptr)(std::forward<ArgTypes>(args)...);
  }

  void set_cuda_dispatch_ptr(FnPtr fn_ptr) {
    impl.cuda_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_hip_dispatch_ptr(FnPtr fn_ptr) {
    impl.hip_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_mps_dispatch_ptr(FnPtr fn_ptr) {
    impl.mps_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  static TORCH_API FnPtr DEFAULT;
#ifdef HAVE_AVX512_CPU_DEFINITION
  static TORCH_API FnPtr AVX512;
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  static TORCH_API FnPtr AVX2;
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  static TORCH_API FnPtr VSX;
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  static TORCH_API FnPtr ZVECTOR;
#endif
private:
  DispatchStubImpl impl;
};

namespace {
template <typename DispatchStub>
struct RegisterCUDADispatch {
  RegisterCUDADispatch(DispatchStub &stub, typename DispatchStub::FnPtr value) {
    stub.set_cuda_dispatch_ptr(value);
  }
};

template <typename DispatchStub>
struct RegisterMPSDispatch {
  RegisterMPSDispatch(DispatchStub &stub, typename DispatchStub::FnPtr value) {
    stub.set_mps_dispatch_ptr(value);
  }
};

template <typename DispatchStub>
struct RegisterHIPDispatch {
  RegisterHIPDispatch(DispatchStub &stub, typename DispatchStub::FnPtr value) {
    // TODO: make this point at hip_dispatch_ptr
    stub.set_cuda_dispatch_ptr(value);
  }
};

} // anonymous namespace
// Compiler will complain if you put things like std::tuple<Tensor, Tensor> in
// the `fn` argument of DECLARE_DISPATCH. Some possible workarounds, e.g.,
// adding parentheses and using helper struct to get rid of the parentheses, do
// not work with MSVC. So do a `using`-declaration if you need to pass in such
// `fn`, e.g., grid_sampler_2d_backward_cpu_kernel in GridSampleKernel.h.
#define DECLARE_DISPATCH(fn, name)         \
  struct name : DispatchStub<fn, name> {   \
    name() = default;                      \
    name(const name&) = delete;            \
    name& operator=(const name&) = delete; \
  };                                       \
  extern TORCH_API struct name name

#define DEFINE_DISPATCH(name) struct name name

#define REGISTER_ARCH_DISPATCH(name, arch, fn) \
  template <> name::FnPtr TORCH_API DispatchStub<name::FnPtr, struct name>::arch = fn;

#ifdef HAVE_AVX512_CPU_DEFINITION
#define REGISTER_AVX512_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, AVX512, fn)
#else
#define REGISTER_AVX512_DISPATCH(name, fn)
#endif

#ifdef HAVE_AVX2_CPU_DEFINITION
#define REGISTER_AVX2_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, AVX2, fn)
#else
#define REGISTER_AVX2_DISPATCH(name, fn)
#endif

#ifdef HAVE_VSX_CPU_DEFINITION
#define REGISTER_VSX_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, VSX, fn)
#else
#define REGISTER_VSX_DISPATCH(name, fn)
#endif

#ifdef HAVE_ZVECTOR_CPU_DEFINITION
#define REGISTER_ZVECTOR_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, ZVECTOR, fn)
#else
#define REGISTER_ZVECTOR_DISPATCH(name, fn)
#endif

/*
REGISTER_DISPATCH
REGISTER_DISPATCH宏在编译期的时候根据device的不同自动展开不同的注册方法。以CPU的设备为例，REGISTER_DISPATCH会展开为：

REGISTER_ARCH_DISPATCH，并随着cmake系统分三次展开（三个编译单元），分别是default、avx、avx2,如下所示：

REGISTER_ARCH_DISPATCH(name, DEFAULT, fn)
REGISTER_ARCH_DISPATCH(name, AVX, fn)
REGISTER_ARCH_DISPATCH(name, AVX2, fn)
#展开为
DispatchStub<decltype(fn), struct name>::DEFAULT = *_kernel;
DispatchStub<decltype(fn), struct name>::AVX = *_kernel;
DispatchStub<decltype(fn), struct name>::AVX2 = *_kernel;
在运行的时候，通过choose_cpu_impl()依次判断pytorch是否使用AVX2、AVX编译的，如果都不是则使用DEFAULT。

另外，上述*_kernel函数对应的有以下几十个：

REGISTER_DISPATCH(pdist_forward_stub, &pdist_forward_kernel_impl);
。。。。。。。
REGISTER_DISPATCH(tanh_stub, &tanh_kernel)
REGISTER_DISPATCH(trunc_stub, &trunc_kernel)
以 REGISTER_DISPATCH(add_stub, &add_kernel) 为例，在CPU设备上的AVX2编译单元中，会展开为如下形式：

template <> decltype(&add_kernel) DispatchStub<decltype(&add_kernel), struct add_stub>::AVX2 = &add_kernel;
从上面的展开式中可以看到注册了add_kernel函数，这个函数实现在了aten/src/ATen/native/cpu/BinaryOpsKernel.cpp文件中。

在PyTorch的运行中，tensor之间的加法会调用到add_stub，并被分发到上述定义的add_kernel函数上：

void add_kernel(TensorIterator& iter, Scalar alpha_scalar) {
  std::cout<<"gemfield call "<<__FILE__<<":"<<__LINE__<<":"<<__FUNCTION__<<std::endl;
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "add_cpu", [&]() {
    auto alpha = alpha_scalar.to<scalar_t>();
    auto alpha_vec = Vec256<scalar_t>(alpha);
    binary_kernel_vec(iter,
      [=](scalar_t a, scalar_t b) -> scalar_t { return a + alpha * b; },
      [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
        return vec256::fmadd(b, alpha_vec, a);
      });
  });
}
总结
Gemfield将PyTorch的初始化分为3个阶段，如下所示：

再谈PyTorch的初始化（上）：介绍c++库main/initModule之前的初始化工作，主要就是global constructors；

再谈PyTorch的初始化（中）：介绍c++库main/initModule之后的初始化工作；

再谈PyTorch的初始化（下）：__init__.py中c++库之外的初始化工作。

本篇文章为上篇，在本篇文章中，gemfield主要介绍c++库main/initModule之前的初始化工作，主要就是global constructors，以及class中的static成员（单例设计模式）。共涉及到PyTorch中以下子系统：
yk====很多内容随着代码更新已经不存在！！！
1，Registry系统的初始化；
2，LegacyDeviceTypeInitRegistry的初始化；
3，g_device_type_registry的初始化；
4，内存分配器Allocator的初始化；
5，Type id系统的初始化；
6，注册Tensor类型；
7，device_guard_impl_registry数组的初始化；
8，THVector table的初始化；
9，CopyBytesFunction表的初始化；
10，Caffe2 OpSchemaRegistry的初始化；
11，ATen中Type继承体系的初始化；
12，C10 dispatcher table的初始化；
13，CodeTemplate的global construct；
14，VariableTypeRegistry及at::globalContext的初始化；
15，JIT operator的初始化；
16，REGISTER_DISPATCH。

*/
// Macro to register the same kernel for all CPU arch types. This is useful
// if a kernel does not benefit from being recompiled across different arch types.
#define REGISTER_ALL_CPU_DISPATCH(name, fn)                                    \
  REGISTER_ARCH_DISPATCH(name, DEFAULT, fn)                                    \
  REGISTER_AVX512_DISPATCH(name, fn)                                           \
  REGISTER_AVX2_DISPATCH(name, fn)                                             \
  REGISTER_VSX_DISPATCH(name, fn)                                              \
  REGISTER_ZVECTOR_DISPATCH(name, fn)

#define REGISTER_NO_CPU_DISPATCH(name)                                         \
  REGISTER_ALL_CPU_DISPATCH(name, nullptr)

#define REGISTER_CUDA_DISPATCH(name, fn) \
  static RegisterCUDADispatch<struct name> name ## __register(name, fn);

#define REGISTER_HIP_DISPATCH(name, fn) \
  static RegisterHIPDispatch<struct name> name ## __register(name, fn);

#define REGISTER_MPS_DISPATCH(name, fn) \
  static RegisterMPSDispatch<struct name> name ## __register(name, fn);

// NB: This macro must be used in an actual 'cu' file; if you try using
// it from a 'cpp' file it will not work!
#if defined(__CUDACC__)
#define REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
#elif defined(__HIPCC__)
// TODO: cut this over to HIP dispatch once we stop pretending that CUDA
// is HIP in the PyTorch HIPify build.
#define REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
// #define REGISTER_DISPATCH(name, fn) REGISTER_HIP_DISPATCH(name, fn)
#elif defined(__OBJC__) && defined(USE_MPS)
// NB: this macro must be used from a 'mm' file in order to dispatch a MPS kernel
#define REGISTER_DISPATCH(name, fn) REGISTER_MPS_DISPATCH(name, fn)
#elif defined(CPU_CAPABILITY)
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#define REGISTER_NO_AVX512_DISPATCH(name)       \
  REGISTER_AVX512_DISPATCH(name, nullptr)
#endif


}} // namespace at::native


#if defined(__clang__)
#pragma clang diagnostic pop
#endif

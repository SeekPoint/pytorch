.. _hip-semantics:

HIP (ROCm) semantics
====================

ROCm\ |trade| is AMD’s open source software platform for GPU-accelerated high
performance computing and machine learning. HIP is ROCm's C++ dialect designed
to ease conversion of CUDA applications to portable C++ code. HIP is used when
converting existing CUDA applications like PyTorch to portable C++ and for new
projects that require portability between AMD and NVIDIA.

.. _hip_as_cuda:

HIP Interfaces Reuse the CUDA Interfaces
----------------------------------------

PyTorch for HIP intentionally reuses the existing :mod:`torch.cuda` interfaces.
This helps to accelerate the porting of existing PyTorch code and models because
very few code changes are necessary, if any.

The example from :ref:`cuda-semantics` will work exactly the same for HIP::


0x02 移动模型到GPU
2.1 cuda 操作
CUDA 是NVIDIA公司开发的GPU编程模型，其提供了GPU编程接口，用户可以基于CUDA编程来构建基于GPU计算的应用。
torch.cuda用于设置 cuda 和运行cuda操作。它跟踪当前选定的GPU，默认情况下，用户分配的所有CUDA张量都将在该设备上创建。
用户可以使用 torch.cuda.device 来修改所选设备。
一旦分配了张量，您可以对其执行操作，而不考虑所选设备，PyTorch 会把运行结果与原始张量放在同一设备上。
默认情况下，除了~torch.Tensor.copy_和其他具有类似复制功能的方法（如~torch.Tensor.to和~torch.Tensor.cuda）之外，
不允许跨GPU操作，除非启用对等（peer-to-peer）内存访问。

我们从源码之中找出一个具体示例如下，大家可以看到，张量可以在设备上被创建，操作。


    cuda = torch.device('cuda')     # Default HIP device
    cuda0 = torch.device('cuda:0')  # 'rocm' or 'hip' are not valid, use 'cuda'
    cuda2 = torch.device('cuda:2')  # GPU 2 (these are 0-indexed)

    x = torch.tensor([1., 2.], device=cuda0)
    # x.device is device(type='cuda', index=0)
    y = torch.tensor([1., 2.]).cuda()
    # y.device is device(type='cuda', index=0)

    with torch.cuda.device(1):
        # allocates a tensor on GPU 1
        a = torch.tensor([1., 2.], device=cuda)

        # transfers a tensor from CPU to GPU 1
        b = torch.tensor([1., 2.]).cuda()
        # a.device and b.device are device(type='cuda', index=1)

        # You can also use ``Tensor.to`` to transfer a tensor:
        b2 = torch.tensor([1., 2.]).to(device=cuda)
        # b.device and b2.device are device(type='cuda', index=1)

        c = a + b
        # c.device is device(type='cuda', index=1)

        z = x + y
        # z.device is device(type='cuda', index=0)

        # even within a context, you can specify the device
        # (or give a GPU index to the .cuda call)
        d = torch.randn(2, device=cuda2)
        e = torch.randn(2).to(cuda2)
        f = torch.randn(2).cuda(cuda2)
        # d.device, e.device, and f.device are all device(type='cuda', index=2)

.. _checking_for_hip:

Checking for HIP
----------------

Whether you are using PyTorch for CUDA or HIP, the result of calling
:meth:`~torch.cuda.is_available` will be the same. If you are using a PyTorch
that has been built with GPU support, it will return `True`. If you must check
which version of PyTorch you are using, refer to this example below::

    if torch.cuda.is_available() and torch.version.hip:
        # do something specific for HIP
    elif torch.cuda.is_available() and torch.version.cuda:
        # do something specific for CUDA

.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:

.. _tf32_on_rocm:

TensorFloat-32(TF32) on ROCm
----------------------------

TF32 is not supported on ROCm.

.. _rocm-memory-management:

Memory management
-----------------

PyTorch uses a caching memory allocator to speed up memory allocations. This
allows fast memory deallocation without device synchronizations. However, the
unused memory managed by the allocator will still show as if used in
``rocm-smi``. You can use :meth:`~torch.cuda.memory_allocated` and
:meth:`~torch.cuda.max_memory_allocated` to monitor memory occupied by
tensors, and use :meth:`~torch.cuda.memory_reserved` and
:meth:`~torch.cuda.max_memory_reserved` to monitor the total amount of memory
managed by the caching allocator. Calling :meth:`~torch.cuda.empty_cache`
releases all **unused** cached memory from PyTorch so that those can be used
by other GPU applications. However, the occupied GPU memory by tensors will not
be freed so it can not increase the amount of GPU memory available for PyTorch.

For more advanced users, we offer more comprehensive memory benchmarking via
:meth:`~torch.cuda.memory_stats`. We also offer the capability to capture a
complete snapshot of the memory allocator state via
:meth:`~torch.cuda.memory_snapshot`, which can help you understand the
underlying allocation patterns produced by your code.

To debug memory errors, set
``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` in your environment to disable caching.

.. _hipfft-plan-cache:

hipFFT/rocFFT plan cache
------------------------

Setting the size of the cache for hipFFT/rocFFT plans is not supported.

Refer to CUDA Semantics doc
---------------------------

For any sections not listed here, please refer to the CUDA semantics doc: :ref:`cuda-semantics`

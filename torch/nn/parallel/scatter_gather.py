import torch
from ._functions import Scatter, Gather
import warnings

__all__ = ['scatter', 'scatter_kwargs', 'gather']

def is_namedtuple(obj):
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    warnings.warn("is_namedtuple is deprecated, please use the python checks instead")
    return _is_namedtuple(obj)

def _is_namedtuple(obj):
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )

#从注释中可以知道，tensor 会切分成大致相等的块，然后在给定的GPU之间分配。就是将一个 batch 数据近似等分成更小的 batch。
# 对于其他类型的变量，会根据不同类型进行不同操作，比如调用 scatter_map 对其内部进行递归处理。
def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            # 针对张量会调用Scatter.apply处理
            return Scatter.apply(target_gpus, None, dim, obj)
        if _is_namedtuple(obj):
            # 调用 scatter_map 对其子模块进行递归处理。
            return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            # 调用 scatter_map 对其子模块进行递归处理。
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            # 调用 scatter_map 对其子模块进行递归处理。
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            # 调用 scatter_map 对其子模块进行递归处理。
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None
    return res

def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    # 分发input
    inputs = scatter(inputs, target_gpus, dim) if inputs else []

    # 分发kwargs,'''
    # scatter_kwargs 函数中最重要的就是 scatter 函数，负责将 tensor 分成大概相等的块并将他们分给不同的 GPU。对其他的数据类型，则是复制分散给不同的 GPU 。
    # '''
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []

    # 用空项补齐，这样可以让 inputs 和 kwargs 长度相等
    if len(inputs) < len(kwargs):
        inputs.extend(() for _ in range(len(kwargs) - len(inputs)))
    elif len(kwargs) < len(inputs):
        kwargs.extend({} for _ in range(len(inputs) - len(kwargs)))

    # 返回 tuple
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

#gather 主要是调用 Gather.apply(target_device, dim, *outputs) 完成收集工作。
def gather(outputs, target_device, dim=0):   # target_device 就是 device[0]
    r"""
    Gathers tensors from different GPUs on a specified device.
    Use 'cpu' for CPU to avoid a deprecation warning.
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs) # 调用下面的 Gather
        if out is None:
            return None
        if isinstance(out, dict):
            if not all(len(out) == len(d) for d in outputs):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)((k, gather_map([d[k] for d in outputs]))
                             for k in out)
        if _is_namedtuple(out):
            return type(out)._make(map(gather_map, zip(*outputs)))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res

'''
最后，用一张图形象地看一下 DP Module 究竟怎么执行的，具体看第一行和第三行：

前向传播的时候我们会先用 Scatter 函数将数据从 device[0] 分配并复制到不同的卡，之后用 Replicate 函数将模型从 device[0] 复制到不同的卡，之后各个卡都有了同样的模型和不同的数据，分别调用 forward 计算损失和梯度。

反向传播的时候，我们会将梯度收集到 device[0] 然后在 device[0] 更新参数。


Fig. 3: DP Module流程图，原图见[4]



1.4 分析
负载不均衡
device[0] 负载大一些

通信开销
假设有 
 个 GPU， 完成一次通信需要时间 
 ，那么使用 PS 算法，总共需要花费时间 

单进程
The difference between DistributedDataParallel and DataParallel is: DistributedDataParallel uses multiprocessing where a process is created for each GPU, while DataParallel uses multithreading. By using multiprocessing, each GPU has its dedicated process, this avoids the performance overhead caused by GIL of Python interpreter.

官方文档
Global Interpreter Lock (GIL)全局解释器锁，简单来说就是，一个 Python 进程只能利用一个 CPU kernel，即单核多线程并发时，只能执行一个线程。考虑多核，多核多线程可能出现线程颠簸 (thrashing) 造成资源浪费，所以 Python 想要利用多核最好是多进程。

'''
r""""Contains definitions of the methods used by the _BaseDataLoaderIter to put
fetched tensors into pinned memory.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import collections
import queue

import torch
from . import MP_STATUS_CHECK_INTERVAL
from torch._utils import ExceptionWrapper

'''
    在主进程之中，如果设置了需要pin memory，
    主进程的pin_memory_thread会从worker_result_queue读取数据，
    进行处理（加速CPU和GPU的数据拷贝），
    把结果放入data_queue。
'''
def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    if device == "cuda":
        torch.cuda.set_device(device_id)
    elif device == "xpu":
        torch.xpu.set_device(device_id)  # type: ignore[attr-defined]

    def do_one_step():
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                data = pin_memory(data, device)
            except Exception:
                data = ExceptionWrapper(
                    where="in pin memory thread for device {}".format(device_id))
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        # Make sure that we don't preserve any object from one iteration
        # to the next
        do_one_step()

'''
1. 单进程

在单进程模式下，DataLoader 初始化的进程和取数据的进程是一样的 。因此，数据加载可能会阻止计算。但是，当用于在进程之间共享数据的资源（例如共享内存，文件描述符）有限时，或者当整个数据集很小并且可以完全加载到内存中时，此模式可能是我们首选。此外，单进程加载通常可以显示更多可读的错误跟踪，这对于我们调试代码很有用。

2. 多进程

多进程处理（multi-process）

为了避免在加载数据时阻塞计算，PyTorch 提供了一个简单的开关，只需将参数设置 num_workers 为正整数即可执行多进程数据加载，而设置为 0 时执行单线程数据加载。

在设置多进程模式时，每次 DataLoader 创建 iterator 时（例如，当调用 enumerate(dataloader) 时），都会创建 num_workers 个工作进程。此时dataset, collate_fn, worker_init_fn 都会被传到每个worker中，而每个 worker 都用独立的进程。

对于 map-style 数据，主线程会用 Sampler 产生 indices，并将它们送到 worker 里。因此，shuffle 是在主线程做的。

而对于 iterable-style 数据，因为每个 worker 都有相同的 data 复制样本，并在各个进程里进行不同的操作，以防止每个进程输出的数据是重复的，所以一般会使用 torch.utils.data.get_worker_info() 来进行辅助处理。这里，torch.utils.data.get_worker_info() 会返回 worker 进程的一些信息(如id, dataset, num_workers, seed)，如果在主线程的话返回 None。

注意，通常不建议在多进程加载中返回 CUDA 张量，因为在使用 CUDA 和在多处理中共享 CUDA 张量时存在许多微妙之处（文档中提出：只要接收过程保留张量的副本，就需要发送过程来保留原始张量）。建议采用 pin_memory=True ，以将数据快速传输到支持 CUDA 的 GPU。简而言之，不建议在使用多线程的情况下返回 CUDA 的 Tensor。

3. 锁页内存

首先我们先了解一下锁页内存的概念。

主机中的内存，有两种存在方式，一是锁页，二是不锁页。锁页内存存放的内容在任何情况下都不会与主机的虚拟内存进行交换（注：虚拟内存就是硬盘），而不锁页内存在主机内存不足时，数据会存放在虚拟内存中。主机到 GPU 副本源自固定（页面锁定）内存时，速度要快得多。CPU 张量和存储暴露了一种 pin_memory() 方法，该方法返回对象的副本，并将数据放在固定的区域中。

而显卡中的显存全部是锁页内存！当计算机的内存充足的时候，可以设置 pin_memory=True。设置 pin_memory=True，则意味着生成的 Tensor 数据最开始是属于内存中的锁页内存，这样将内存的 Tensor 转义到 GPU 的显存就会更快一些。同时，由于 pin_memory 的作用是将张量返回之前将其复制到 CUDA 固定的内存中，所以只有在 CUDA 环境支持下才有用。

PyTorch 原生的 pin_memory 方法如下，其支持大部分 python 数据类型的处理：
'''
def pin_memory(data, device=None):
    if isinstance(data, torch.Tensor):
        return data.pin_memory(device)
    elif isinstance(data, str):
        return data
    elif isinstance(data, collections.abc.Mapping):
        try:
            return type(data)({k: pin_memory(sample, device) for k, sample in data.items()})  # type: ignore[call-arg]
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {k: pin_memory(sample, device) for k, sample in data.items()}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(pin_memory(sample, device) for sample in data))
    elif isinstance(data, tuple):
        return [pin_memory(sample, device) for sample in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence):
        try:
            return type(data)([pin_memory(sample, device) for sample in data])  # type: ignore[call-arg]
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [pin_memory(sample, device) for sample in data]
    elif hasattr(data, "pin_memory"):
        return data.pin_memory()
    else:
        return data

import threading
import torch
from torch.cuda._utils import _get_device_index
from torch.cuda.amp import autocast
from torch._utils import ExceptionWrapper


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, (list, tuple)):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None

#parallel_apply 是基于threading 实现，用前面准备好的 replica 和输入数据，然后for 循环启动多线程进行前向传播，最后输出传播结果。
'''
此时前向传播具体对应如下图，现在并行操作调用了 module 的forward方法。

+----------------------------------------------------------------------------------------+
| DataParallel.forward                                                                   |
|                                                                                        |
|                  1                               2                      3              |
|              replicate +--------------->   parallel_apply             gather           |
|                                                                                        |
+----------------------------------------------------------------------------------------+

     +---------------------------+       +-------------------+
     | Broadcast                 |       | module            |
     |                           |       |                   |
     |                           |       |                   |
     |              1            |       |         2         |
     |          forward()  +-----------> |      forward() +--------->
     |                           |       |                   |
     |                           |       |                   |
     |  +---------------------+  |       |                   |
     |  | ctx                 |  |       |                   |
     |  |       input_device  |  |       |                   |
     |  |                     |  |       |                   |
     |  |       num_inputs    |  |       |                   |
     |  |                     |  |       |                   |
     |  +---------------------+  |       |                   |
     |                           |       |                   |
     |                           |       |                   |
     |                           |       |                   |
     |                           |       |                   |
     |                           |       |                   |
     |                           |       |                   |
     +---------------------------+       +-------------------+

'''
def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    # 确保模型和输入大小一致  # 每个 GPU 都有模型和输入
    assert len(modules) == len(inputs)

    # 确保每个 GPU 都有相应的元数据，如没有就空白补全
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)

    # 确保模型数目和CPU数目一致
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = [_get_device_index(x, True) for x in devices]
    streams = [torch.cuda.current_stream(x) for x in devices]

    # 基于threading多线程实现
    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

    # 定义 worker
    def _worker(i, module, input, kwargs, device=None, stream=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        if stream is None:
            stream = torch.cuda.current_stream(device)
        try:
            # 设置当前的设备
            with torch.cuda.device(device), torch.cuda.stream(stream), autocast(enabled=autocast_enabled):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, **kwargs)  # 前向操作
            with lock:
                results[i] = output
        except Exception:
            with lock:
                # 并行计算得到输出
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        # 如有一个进程控制多个 GPU ，起多个线程
        # 注意，这里就是每个 worker 调用了 modules 数组中的一个模型copy
        # 如有一个进程控制多个 GPU ，起多个线程
        # 需要强调一下，虽然 DDP 推荐单卡单进程，即每次调用 DDP device_ids 都只输入一张卡的 id（通常是 args.local_rank），
        # 但是如果输入多个 device_id，此时 DDP 就是单进程多线程控制多卡，和 DP 一样，关于 DDP 的解读可以看下文

        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device, stream))
                   for i, (module, input, kwargs, device, stream) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices, streams))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        # 一个 GPU 一个进程 （ DDP 推荐操作）
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0], streams[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    # 输出 n 个计算结果
    return outputs

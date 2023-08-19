import warnings

import torch
from . import comm
from torch.autograd import Function
from torch._utils import _get_device_index
from typing import List, Optional

#使用 Broadcast 过度一下的原因是：因为张量不是 detached，所以除了广播之外，还需要在上下文中设置哪些不需要梯度。
# 在某些情况下，用户自定义的Function可能需要知道此情况。
#这部分对应了 Broadcast 的 反向传播。
'''
因此，我们可以拓展流程图：

+--------------------------------------------------------------------------------------+
| DataParallel.forward                                                                 |
|                                                                                      |
|               1                               2                           3          |
|           replicate +--------------->   parallel_apply +--------------> gather       |
|                                                                                      |
+--------------------------------------------------------------------------------------+

  +---------------------------+       +-------------------+       +--------------------+
  | Broadcast                 |       | module            |       |Gather              |
  |                           |       |                   |       |                    |
  |                           |       |                   |       |                    |
  |              1            |       |         2         |       |         3          |
  |          forward()  +-----------> |      forward() +--------> |      forward()     |
  |                           |       |                   |       |                    |
  |                           |       |                   |       |                    |
  |  +---------------------+  |       |                   |       | +----------------+ |
  |  | ctx                 |  |       |                   |       | |ctx             | |
  |  |       input_device  |  |       |                   |       | |     input_gpus | |
  |  |                     |  |       |                   |       | |                | |
  |  |       num_inputs    |  |       |                   |       | |     input_sizes| |
  |  |                     |  |       |                   |       | |                | |
  |  +---------------------+  |       |                   |       | |     dim        | |
  |                           |       |                   |       | +----------------+ |
  |                           |       |                   |       |                    |
  |                           |       |                   |       |                    |
  |          backward()       | <---------+  backward()   | <---------+ backward()     |
  |              5            |       |          4        |       |         3          |
  |                           |       |                   |       |                    |
  +---------------------------+       +-------------------+       +--------------------+

+--------------------------------------------------------------------------------------+
| loss.backward()                                                                      |
|                5                               4                          3          |
|         <------------------------+  <------------------+  <--------------------+     |
|                                                                                      |
|                                                                                      |
+--------------------------------------------------------------------------------------+

'''
#  Broadcast.apply
class Broadcast(Function):

    @staticmethod
    def forward(ctx, target_gpus, *inputs):
        assert all(i.device.type != 'cpu' for i in inputs), (
            'Broadcast function not implemented for CPU tensors'
        )
        target_gpus = [_get_device_index(x, True) for x in target_gpus]

        # 前向传播时候，向上下文存入了一些变量
        ctx.target_gpus = target_gpus
        if len(inputs) == 0:
            return tuple()
        ctx.num_inputs = len(inputs)

        # input 放在 device[0]  # input 放在 device[0]，所以 input_device 就是 GPU 0
        ctx.input_device = inputs[0].get_device()

        # 和 detach 的情形一样
        outputs = comm.broadcast_coalesced(inputs, ctx.target_gpus)
        non_differentiables = []

        # 在上下文中设置哪些不需要梯度
        for idx, input_requires_grad in enumerate(ctx.needs_input_grad[1:]):
            if not input_requires_grad:
                for output in outputs:
                    non_differentiables.append(output[idx])
        ctx.mark_non_differentiable(*non_differentiables)
        return tuple([t for tensors in outputs for t in tensors])

    @staticmethod
    def backward(ctx, *grad_outputs):
        # 反向传播来到这里，取出之前在上下文存放的变量作为输入。ctx.input_device 就是之前存储的 GPU 0。
        return (None,) + ReduceAddCoalesced.apply(ctx.input_device, ctx.num_inputs, *grad_outputs)

#Broadcast.backward 调用了 ReduceAddCoalesced.apply，其对应了 ReduceAddCoalesced 的 forward 方法，目的是把梯度归并到目标设备 destination，就是GPU 0。
class ReduceAddCoalesced(Function):

    @staticmethod
    # 会调用到这里，destination 是GPU 0
    def forward(ctx, destination, num_inputs, *grads):
        # 从梯度之中提取所在的设备
        ctx.target_gpus = [grads[i].get_device() for i in range(0, len(grads), num_inputs)]

        grads_ = [grads[i:i + num_inputs]
                  for i in range(0, len(grads), num_inputs)]
        # 把梯度归并到目标设备 destination，就是GPU 0
        return comm.reduce_add_coalesced(grads_, destination)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None,) + Broadcast.apply(ctx.target_gpus, *grad_outputs)

#Gather 则调用了 comm.gather 完成工作，而 comm.gather 则会带领我们进入到 C++世界。
'''
现在前向计算如图：

gather 调用到了Gather的forward 函数，forward 方法在 ctx 存储了 input_gpus, input_sizes, dim 这三个变量，这些变量后续会用到。

+-----------------------------------------------------------------------------------------+
| DataParallel.forward                                                                    |
|                                                                                         |
|                  1                               2                           3          |
|              replicate +--------------->   parallel_apply +--------------> gather       |
|                                                                                         |
+-----------------------------------------------------------------------------------------+

     +---------------------------+       +-------------------+       +--------------------+
     | Broadcast                 |       | module            |       |Gather              |
     |                           |       |                   |       |                    |
     |                           |       |                   |       |                    |
     |              1            |       |         2         |       |         3          |
     |          forward()  +-----------> |      forward() +--------> |      forward()     |
     |                           |       |                   |       |                    |
     |                           |       |                   |       |                    |
     |  +---------------------+  |       |                   |       | +----------------+ |
     |  | ctx                 |  |       |                   |       | |ctx             | |
     |  |       input_device  |  |       |                   |       | |     input_gpus | |
     |  |                     |  |       |                   |       | |                | |
     |  |       num_inputs    |  |       |                   |       | |     input_sizes| |
     |  |                     |  |       |                   |       | |                | |
     |  +---------------------+  |       |                   |       | |     dim        | |
     |                           |       |                   |       | +----------------+ |
     |                           |       |                   |       |                    |
     |                           |       |                   |       |                    |
     |                           |       |                   |       |                    |
     |                           |       |                   |       |                    |
     |                           |       |                   |       |                    |
     +---------------------------+       +-------------------+       +--------------------+




具体如下，可以看到，backward 使用了之前前向传播时候存储的 ctx.input_gpus, ctx.input_sizes, ctx.dim, grad_output，以此调用 Scatter.apply。

图中，最上面是前向传播过程，最下面是反向传播过程，中间是某些在前后传播中都用到的代码模块。

+--------------------------------------------------------------------------------------+
| DataParallel.forward                                                                 |
|                                                                                      |
|               1                               2                           3          |
|           replicate +--------------->   parallel_apply +--------------> gather       |
|                                                                                      |
+--------------------------------------------------------------------------------------+

  +---------------------------+       +-------------------+       +--------------------+
  | Broadcast                 |       | module            |       |Gather              |
  |                           |       |                   |       |                    |
  |                           |       |                   |       |                    |
  |              1            |       |         2         |       |         3          |
  |          forward()  +-----------> |      forward() +--------> |      forward()     |
  |                           |       |                   |       |                    |
  |                           |       |                   |       |                    |
  |  +---------------------+  |       |                   |       | +----------------+ |
  |  | ctx                 |  |       |                   |       | |ctx             | |
  |  |       input_device  |  |       |                   |       | |     input_gpus | |
  |  |                     |  |       |                   |       | |                | |
  |  |       num_inputs    |  |       |                   |       | |     input_sizes| |
  |  |                     |  |       |                   |       | |                | |
  |  +---------------------+  |       |                   |       | |     dim        | |
  |                           |       |                   |       | +----------------+ |
  |                           |       |                   |       |                    |
  |                           |       |                   |       |                    |
  |                           |       |                   | <---------+ backward()     |
  |                           |       |                   |       |         3          |
  |                           |       |                   |       |                    |
  +---------------------------+       +-------------------+       +--------------------+

+--------------------------------------------------------------------------------------+
| loss.backward()                                                                      |
|                                                                           3          |
|                                                           <--------------------+     |
|                                                                                      |
|                                                                                      |
+--------------------------------------------------------------------------------------+

'''
class Gather(Function):

    @staticmethod
    def forward(ctx, target_device, dim, *inputs):    # target_device 就是 device[0]
        assert all(i.device.type != 'cpu' for i in inputs), (
            'Gather function not implemented for CPU tensors'
        )

        # 下面会往 context 内部存放几个变量，后续会用到
        if (target_device == 'cpu'):
            ctx.target_device = 'cpu'
        else:
            target_device = _get_device_index(target_device, True)
            ctx.target_device = target_device
        ctx.dim = dim
        ctx.input_gpus = tuple(i.get_device() for i in inputs)
        if all(t.dim() == 0 for t in inputs) and dim == 0:
            inputs = tuple(t.view(1) for t in inputs)
            warnings.warn('Was asked to gather along dimension 0, but all '
                          'input tensors were scalars; will instead unsqueeze '
                          'and return a vector.')
            ctx.unsqueezed_scalar = True
        else:
            ctx.unsqueezed_scalar = False
        ctx.input_sizes = tuple(i.size(ctx.dim) for i in inputs)
        return comm.gather(inputs, ctx.dim, ctx.target_device)  # # 这里会进入C++世界，把输出聚集到 GPU 0。

    @staticmethod
    def backward(ctx, grad_output):  # 注意，# 这里现在后向传播用到了！
        # 把前向传播在 context 之中存放的变量取出，作为 Scatter 的输入
        scattered_grads = Scatter.apply(ctx.input_gpus, ctx.input_sizes, ctx.dim, grad_output)
        if ctx.unsqueezed_scalar:
            scattered_grads = tuple(g[0] for g in scattered_grads)
        return (None, None) + scattered_grads

'''
前面提到了 Scatter.apply 处理张量，我们就接着看看。Scatter 拓展了 Function，逻辑如下：

    如果 cuda 可用，则得到 streams 列表，这样可以在后台流进行 CPU 到 GPU 的拷贝。
    调用 comm.scatter 进行分发。
    调用 wait_stream 和 record_stream 对拷贝流进行同步。
    

Scatter.apply 实际上调用到了其 forward 方法。

首先从上下文之中提取之前存储的变量，这里主要是输入设备 input_device（源设备）和 target_gpus（目标设备）。
获取到目标设备的流。
调用 comm.scatter 把梯度分发到目标设备。
'''
class Scatter(Function):

    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input):
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.dim = dim
        ctx.input_device = input.get_device() if input.device.type != "cpu" else -1
        streams = None
        # 对于cuda，进行处理
        if torch.cuda.is_available() and ctx.input_device == -1:
            # Perform CPU to GPU copies in a background stream
            # 新建 cuda stream
            streams = [_get_stream(device) for device in target_gpus]

        # 调用C++进行操作  # 分发到其他GPU  会直接进入到C++世界  具体代码位于 torch/csrc/cuda/comm.cpp。
        outputs = comm.scatter(input, target_gpus, chunk_sizes, ctx.dim, streams)

        # Synchronize with the copy stream
        if streams is not None:
            for i, output in enumerate(outputs):
                with torch.cuda.device(target_gpus[i]):
                    main_stream = torch.cuda.current_stream()
                    main_stream.wait_stream(streams[i])  # 同步
                    output.record_stream(main_stream)   # 同步
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, Gather.apply(ctx.input_device, ctx.dim, *grad_output)


# background streams used for copying
_streams: Optional[List[Optional[torch.cuda.Stream]]] = None


def _get_stream(device: int):
    """Gets a background stream for copying between CPU and GPU"""
    global _streams
    if device == -1:
        return None
    if _streams is None:
        _streams = [None] * torch.cuda.device_count()
    if _streams[device] is None:
        _streams[device] = torch.cuda.Stream(device)
    return _streams[device]

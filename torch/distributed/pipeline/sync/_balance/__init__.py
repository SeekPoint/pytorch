# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""A helper to roughly balance a sequential module.

Usage::

    import torch
    from torch.distributed.pipeline.sync import Pipe
    from torch.distributed.pipeline.sync.balance import balance_by_time

    sample = torch.empty(128, 3, 224, 224)
    balance = balance_by_time(torch.cuda.device_count(), model, sample)

    pipe = Pipe(model, balance, chunks=8)

"""
from typing import Any, List, Union, Sequence

import torch
from torch import Tensor
import torch.nn as nn

from . import blockpartition
from .profile import profile_sizes, profile_times

__all__ = ["balance_by_time", "balance_by_size"]


Device = Union[torch.device, int, str]

Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]

#具体 balance_cost 只是一个封装而已，算法还是 blockpartition.solve。
def balance_cost(cost: List[int], partitions: int) -> List[int]:
    partitioned = blockpartition.solve(cost, partitions)
    return [len(p) for p in partitioned]

'''
1.3 据计算时间来平衡
balance_by_time 方法的作用就是依据运行时间来平衡，其中参数如下：
    partitions ：分区数目
    module : 需要分区的顺序模型
    sample ：给定 batch size 的样本
其实就是调用 profile_times 依据sample来得到运行时间，然后进行分区。

这里的 Batch 类就是对张量或者张量数组进行封装，可以统一使用其方法。
'''
def balance_by_time(
    partitions: int,
    module: nn.Sequential,
    sample: Union[List[Any], Tensor],
    *,
    timeout: float = 1.0,
    device: Device = torch.device("cuda"),
) -> List[int]:
    """Naive automatic balancing by elapsed time per layer.
    ::

        sample = torch.empty(128, 3, 224, 224)
        balance = balance_by_time(torch.cuda.device_count(), model, sample)
        pipe = Pipe(model, balance, chunks=8)

    Args:
        partitions (int):
            intended number of partitions
        module (torch.nn.Sequential):
            sequential module to be partitioned
        sample (torch.Tensor):
            example input with arbitrary batch size

    Keyword Args:
        timeout (float):
            profiling iterates again if the timeout (in second) is not exceeded
            (default: ``1.0``)
        device ('cpu' or 'cuda' device):
            CPU or CUDA device where each layer is profiled (default: the
            current CUDA device)

    Returns:
        A list of number of layers in each partition. Use it for the `balance`
        parameter of :class:`~torchpipe.Pipe`.

    .. note::
        `module` and `sample` must be placed on the same device.

    """
    '''
    1.5 分割算法
    得到每层的计算时间或者内存大小之后，会通过如下代码来进行具体分割。
    '''
    times = profile_times(module, sample, timeout, torch.device(device))
    return balance_cost(times, partitions)

'''
1.4 据内存大小来平衡
balance_by_size 方法的作用就是依据运行时内存大小来平衡，其中参数如下：

    partitions ：分区数目，从示例看，可以认为是设备数。
    module : 需要分区的顺序模型
    sample ：给定 batch size 的样本

其实就是调用 profile_sizes 依据sample来得到运行时内存大小，然后进行分区。
在训练期间，参数所需的内存取决于使用哪个优化器。优化器可以为每个参数使用缓冲区来在其内部跟踪优化统计信息，例如SGD中的动量缓冲区。
为了获得更可靠的基于大小的平衡，用户应该为优化器指定相应的“param_scale”。默认的“param_scale”是2，而不是1，这是因为梯度累积（gradient accumulation）是每个优化器所必需的。下面注释之中也给出了一些参考取值。
'''
def balance_by_size(
    partitions: int,
    module: nn.Sequential,
    input: Union[List[Any], Tensor],
    *,
    chunks: int = 1,
    param_scale: float = 2.0,
    device: Device = torch.device("cuda"),
) -> List[int]:
    """Naive automatic balancing by CUDA memory usage per layer.

    During training, required memory for parameters depends on which optimizer
    is used. Optimizers may use buffers for each parameter to track
    optimization statistics internally, such as momentum buffer in SGD.

    To get more reliable size based balance, you should specify `param_scale`
    with regard to your optimizer. The default `param_scale` is 2 instead of 1
    due to gradient accumulation which is necessary for every optimizer.

    Follow this guide to choose correct `param_scale` for typical optimizers:

    =========  =============  =========================================
    Optimizer  `param_scale`  Internal State
    =========  =============  =========================================
    SGD        2--3           (momentum_buffer)
    Adam       4--5           exp_avg, exp_avg_sq, (max_exp_avg_sq)
    Adadelta   4              square_avg, acc_delta
    Adagrad    3              sum
    RMSprop    3--5           square_avg, (momentum_buffer), (grad_avg)
    =========  =============  =========================================

    Here's a simple example with the Adam optimizer::

        balance = balance_by_size(
            torch.cuda.device_count(),
            model,

            # Same size with mini-batch to train
            torch.empty(1024, 3, 224, 224),

            # Number of micro-batches to train with Pipe
            chunks=8,

            # 4 for Adam
            param_scale=4.0,
        )

        pipe = Pipe(model, balance, chunks=8)
        adam = Adam(pipe.parameters())

    Args:
        partitions (int):
            intended number of partitions
        module (torch.nn.Sequential):
            sequential module to be partitioned
        input (torch.Tensor):
            example mini-batch with the same size to train

    Keyword Args:
        chunks (int):
            number of micro-batches will be used to train (default: ``1``)
        param_scale (float):
            how many copies of parameters would be allocated for training. It
            depends on optimizer. See the above guide. (default: ``2.0``)
        device ('cuda' device):
            CUDA device where each layer is profiled (default: the current CUDA
            device)

    Returns:
        A list of number of layers in each partition. Use it for the `balance`
        parameter of :class:`~torchpipe.Pipe`.

    .. note::
        `module` and `input` must be placed on the same CUDA device.

    """
    sizes = profile_sizes(module, input, chunks, param_scale, torch.device(device))
    return balance_cost(sizes, partitions)

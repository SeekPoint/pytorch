# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Per-layer profilers."""
import copy
import time
from typing import Generator, List, Union, Sequence

import torch
from torch import Tensor
import torch.nn as nn

from ..microbatch import Batch

__all__: List[str] = []


Device = Union[torch.device, int, str]

Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]

# 1.2.2 layerwise_sandbox
# layerwise_sandbox 方法的作用是在不影响原有模型的基础上，拷贝模型的层，这样更容易profile。
def layerwise_sandbox(module: nn.Sequential, device: torch.device,) -> Generator[nn.Module, None, None]:
    """Copies layers for ease to profile. It doesn't modify the given
    module.
    """
    for layer in module:
        layer_copy = copy.deepcopy(layer)
        layer_copy.to(device)
        layer_copy.train()
        yield layer_copy

# 1.2.3 detach
# detach 方法的作用是从autograd图中detach一些张量，得到一组新的张量。
# 这些张量从当前计算图中被分离下来。但是仍指向原变量的存放位置。detach 可以切断一些分支的反向传播.。
def detach(batch: Batch) -> None:
    """Detaches from autograd graph."""
    for i, x in enumerate(batch):
        batch[i] = x.detach().requires_grad_(x.requires_grad)

# profile_times 依据sample来得到运行时间，具体逻辑是：
#     遍历模型中的层，针对每个层：
#         等待当前设备上所有流中的所有kernel完成
#         记录起始运行时间
#         对某层进行前向计算
#         得到需要梯度的张量，如果存在，则进行后向计算
#         等待当前设备上所有流中的所有kernel完成
#         记录终止时间
#     最后返回一个每层运行时间列表。
def profile_times(module: nn.Sequential, sample: TensorOrTensors, timeout: float, device: torch.device,) -> List[int]:
    """Profiles elapsed times per layer."""
    if any(p.grad is not None for p in module.parameters()):
        raise ValueError("some parameter already has gradient")

    _batch = Batch(sample)
    for i, x in enumerate(_batch):
        _batch[i] = x.detach().to(device).requires_grad_(x.requires_grad)

    time_bufs: List[List[float]] = [[] for _ in module]
    begun_at = time.time()

    while time.time() - begun_at < timeout:
        batch = _batch

        # 遍历模型中的层
        for i, layer in enumerate(layerwise_sandbox(module, device)):
            detach(batch)

            if device.type == "cuda":
                torch.cuda.synchronize(device)   # 等待当前设备上所有流中的所有kernel完成
            tick = time.time()  # 起始运行时间

            # Forward
            batch = batch.call(layer)  # 对某层进行前向计算

            # Backward
            # 得到需要梯度的张量
            backward_tensors = tuple(y for y in batch if y.requires_grad)
            if backward_tensors:
                torch.autograd.backward(backward_tensors, backward_tensors)

            if device.type == "cuda":
                torch.cuda.synchronize(device)   # 等待当前设备上所有流中的所有kernel完成
            tock = time.time()  # 终止时间

            time_bufs[i].append(tock - tick)

    us = 1_000_000
    return [sum(int(t * us) for t in buf) for buf in time_bufs]

# profile_sizes 逻辑如下：
#     遍历模型中的层，针对每个层：
#         使用 torch.cuda.memory_allocated 计算前向传播用到的显存，就是激活值。
#         torch.cuda.memory_allocated(device=None) 返回给定设备device的张量所占用的当前GPU内存。
#
#         使用 p.storage().size() * p.storage().element_size() 计算参数尺寸。
#             pytorch中的storage指的是连续的内存块，而tensor可以认为是映射到storage的视图。
#             element_size() 返回单个元素的字节。
#
#         把激活值和参数加在一起，插入列表。
#
#     返回内存大小列表。
def profile_sizes(
    module: nn.Sequential, input: TensorOrTensors, chunks: int, param_scale: float, device: torch.device,
) -> List[int]:
    """Profiles CUDA memory usage per layer."""
    if device.type != "cuda":
        raise ValueError("size profiler supports only CUDA device")

    batch = Batch(input)
    sizes: List[int] = []

    latent_scale = batch[0].size(0) / chunks
    for i, x in enumerate(batch):
        batch[i] = x[:1].detach().to(device).requires_grad_(x.requires_grad)

    for layer in layerwise_sandbox(module, device):
        detach(batch)

        # Detect memory usage at forward.
        # 计算前向传播用到的显存，就是激活值
        memory_before = torch.cuda.memory_allocated(device)
        batch = batch.call(layer)  # 对某层进行前向传播
        memory_after = torch.cuda.memory_allocated(device)
        latent_size = memory_after - memory_before

        # Analyze size of parameters.
        # 计算参数尺寸
        param_size = sum(p.storage().size() * p.storage().element_size() for p in layer.parameters())

        # 把激活值和参数加在一起，插入列表
        # Combine size of parameters and activations with normalize scales.
        size = latent_size * latent_scale + param_size * param_scale
        sizes.append(int(size))

    return sizes  # 返回内存大小列表

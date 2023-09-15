# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Arbitrary dependency between two autograd lanes."""
from typing import List, Tuple

import torch
from torch import Tensor

from .phony import get_phony

__all__: List[str] = []

'''
2.2 基础功能
2.2.1 Function
首先，我们要看看 torch.autograd.Function 的作用。

torch.autograd.Function类实际上是一个操作函数的基础父类，这样的操作函数必须具备两个基本的过程，即前向的运算过程和反向的求导过程，

如果某些操作无法通过 PyTorch 已有的层或者是已有的方法实现不了，就需要实现一个新的方法对 PyTorch 进行拓展。
当不使用自动求导机制，需要自定义求导规则的时候，就应该拓展torch.autograd.Function类。 
由于pytorch不再提供自动求导机制，就要用户自己定义实现前向传播和反向传播的计算过程，这就是 "Extending torch.autograd"。

我们接下来介绍Backward Dependency 的关键算法：Fork and Join。

2.2.2 Fork
Fork 是auto grad 函数，其把一个张量 x 映射到 pair(x, ϕ)，这里 ϕ 是一个空张量。Fork 方法就是拓展了torch.autograd.Function。
'''
def fork(input: Tensor) -> Tuple[Tensor, Tensor]:
    """Branches out from an autograd lane of the given tensor."""
    if torch.is_grad_enabled() and input.requires_grad:
        input, phony = Fork.apply(input)
    else:
        phony = get_phony(input.device, requires_grad=False)

    return input, phony


class Fork(torch.autograd.Function):
    @staticmethod
    def forward(ctx: "Fork", input: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        phony = get_phony(input.device, requires_grad=False)
        return input.detach(), phony.detach()

    @staticmethod
    def backward(ctx: "Fork", grad_input: Tensor, grad_grad: Tensor) -> Tensor:  # type: ignore[override]
        return grad_input

#2.2.3 Join
# Join 是auto grad 函数，其把 pair(x, ϕ) 映射到一个张量 x ，这里 ϕ是一个空张量。Join 方法也是拓展了torch.autograd.Function。
def join(input: Tensor, phony: Tensor) -> Tensor:
    """Merges two autograd lanes."""
    if torch.is_grad_enabled() and (input.requires_grad or phony.requires_grad):
        input = Join.apply(input, phony)

    return input


class Join(torch.autograd.Function):
    @staticmethod
    def forward(ctx: "Join", input: Tensor, phony: Tensor) -> Tensor:  # type: ignore[override]
        return input.detach()

    @staticmethod
    def backward(ctx: "Join", grad_input: Tensor) -> Tuple[Tensor, None]:  # type: ignore[override]
        return grad_input, None

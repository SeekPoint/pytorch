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

__all__: List[str] = ["fork", "Fork", "join", "Join"]

#Fork 是auto grad 函数，其把一个张量 x 映射到 pair(x, ϕ)，这里 ϕ 是一个空张量。
# Fork 方法就是拓展了torch.autograd.Function。
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

#Join 是auto grad 函数，其把 pair(x, ϕ) 映射到一个张量 x ，这里 ϕ 是一个空张量。Join 方法也是拓展了torch.autograd.Function。
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
